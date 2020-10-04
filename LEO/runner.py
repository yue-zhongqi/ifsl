# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A binary building the graph and performing the optimization of LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.INFO)

from absl import flags
import ifsl_configs
from six.moves import zip
import tensorflow as tf

import config
import data
import model
import utils
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "/data2/yuezhongqi/Model/leo/ifsl/miniresnet5baselinewoaug", "Path to restore from and "
                    "save to checkpoints.")
flags.DEFINE_integer(
    "checkpoint_steps", 1000, "The frequency, in number of "
    "steps, of saving the checkpoints.")
flags.DEFINE_boolean("evaluation_mode", False, "Whether to run in an "
                     "evaluation-only mode.")


def _clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
  """Clips gradients by value and then by norm."""
  if gradient_threshold > 0:
    gradients = [
        tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
        for g in gradients
    ]
  if gradient_norm_threshold > 0:
    gradients = [
        tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
    ]
  return gradients


def _construct_validation_summaries(metavalid_loss, metavalid_accuracy):
  tf.summary.scalar("metavalid_loss", metavalid_loss)
  tf.summary.scalar("metavalid_valid_accuracy", metavalid_accuracy)
  # The summaries are passed implicitly by TensorFlow.


def _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  model_grads, model_vars):
  tf.summary.scalar("metatrain_loss", metatrain_loss)
  tf.summary.scalar("metatrain_valid_accuracy", metatrain_accuracy)
  for g, v in zip(model_grads, model_vars):
    histogram_name = v.name.split(":")[0]
    tf.summary.histogram(histogram_name, v)
    histogram_name = "gradient/{}".format(histogram_name)
    tf.summary.histogram(histogram_name, g)


def _construct_examples_batch(batch_size, split, num_classes,
                              num_tr_examples_per_class,
                              num_val_examples_per_class,
                              use_cross=False):
  data_provider = data.DataProvider(split, config.get_data_config(), feat_dim=FLAGS.feat_dim, use_cross=use_cross)
  examples_batch = data_provider.get_batch(batch_size, num_classes,
                                           num_tr_examples_per_class,
                                           num_val_examples_per_class,
                                           num_pretrain_classes=FLAGS.num_pretrain_classes)
  return utils.unpack_data(examples_batch)


def _construct_loss_and_accuracy(inner_model, inputs, is_meta_training):
  """Returns batched loss and accuracy of the model ran on the inputs."""
  call_fn = functools.partial(
      inner_model.__call__, is_meta_training=is_meta_training)
  per_instance_loss, per_instance_accuracy, per_instance_dacc = tf.map_fn(
      call_fn,
      inputs,
      dtype=(tf.float32, tf.float32, tf.float32),
      back_prop=is_meta_training)
  loss = tf.reduce_mean(per_instance_loss)
  accuracy = tf.reduce_mean(per_instance_accuracy)
  dacc = tf.reduce_mean(per_instance_dacc)
  return loss, accuracy, dacc


def construct_debug_graph(outer_model_config):
  inner_model_config = config.get_inner_model_config()
  tf.logging.info("inner_model_config: {}".format(inner_model_config))
  # leo = model.LEO(inner_model_config, use_64bits_dtype=False)
  ifsl = model.IFSL(inner_model_config, use_64bits_dtype=False, n_splits=4)
  num_classes = outer_model_config["num_classes"]
  num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]
  # Construct a batch from training
  metatrain_batch = _construct_examples_batch(
      outer_model_config["metatrain_batch_size"], "train", num_classes,
      num_tr_examples_per_class,
      outer_model_config["num_val_examples_per_class"])
  # call_fn = functools.partial(ifsl.__call__, is_meta_training=True)
  # losses, accuracies = tf.map_fn(call_fn, metatrain_batch, dtype=(tf.float32, tf.float32), back_prop=True)
  losses, accuracies, outputs = ifsl(metatrain_batch, True)
  global_step = tf.train.get_or_create_global_step()
  return losses, accuracies, outputs, metatrain_batch, global_step


def construct_graph(outer_model_config):
  """Constructs the optimization graph."""
  inner_model_config = config.get_inner_model_config()
  tf.logging.info("inner_model_config: {}".format(inner_model_config))
  if FLAGS.deconfound:
    leo = model.IFSL(inner_model_config, use_64bits_dtype=False, n_splits=FLAGS.n_splits,
                     is_cosine_feature=FLAGS.is_cosine_feature, fusion=FLAGS.fusion,
                     classifier=FLAGS.classifier, num_classes=FLAGS.pretrain_num_classes,
                     logit_fusion=FLAGS.logit_fusion, use_x_only=FLAGS.use_x_only,
                     preprocess_before_split=FLAGS.preprocess_before_split,
                     preprocess_after_split=FLAGS.preprocess_after_split,
                     normalize_before_center=FLAGS.normalize_before_center,
                     normalize_d=FLAGS.normalize_d, normalize_ed=FLAGS.normalize_ed)
  else:
    # leo = model.LEO(inner_model_config, use_64bits_dtype=False)
    leo = model.IFSL(inner_model_config, False, 1, False, "concat", "single", FLAGS.pretrain_num_classes,
                      "product", True)

  num_classes = outer_model_config["num_classes"]
  num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]
  metatrain_batch = _construct_examples_batch(
      outer_model_config["metatrain_batch_size"], "train", num_classes,
      num_tr_examples_per_class,
      outer_model_config["num_val_examples_per_class"])
  metatrain_loss, metatrain_accuracy, metatrain_dacc = _construct_loss_and_accuracy(
      leo, metatrain_batch, True)

  metatrain_gradients, metatrain_variables = leo.grads_and_vars(metatrain_loss)

  # Avoids NaNs in summaries.
  metatrain_loss = tf.cond(tf.is_nan(metatrain_loss),
                           lambda: tf.zeros_like(metatrain_loss),
                           lambda: metatrain_loss)

  metatrain_gradients = _clip_gradients(
      metatrain_gradients, outer_model_config["gradient_threshold"],
      outer_model_config["gradient_norm_threshold"])

  _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                metatrain_gradients, metatrain_variables)
  optimizer = tf.train.AdamOptimizer(
      learning_rate=outer_model_config["outer_lr"])
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(
      list(zip(metatrain_gradients, metatrain_variables)), global_step)

  data_config = config.get_data_config()
  tf.logging.info("data_config: {}".format(data_config))
  total_examples_per_class = data_config["total_examples_per_class"]

  split = "val"
  metavalid_batch = _construct_examples_batch(
      outer_model_config["metavalid_batch_size"], split, num_classes,
      num_tr_examples_per_class,
      total_examples_per_class - num_tr_examples_per_class)
  metavalid_loss, metavalid_accuracy, metavalid_dacc = _construct_loss_and_accuracy(
      leo, metavalid_batch, False)

  if not FLAGS.cross:
    metatest_batch = _construct_examples_batch(
        outer_model_config["metatest_batch_size"], "test", num_classes,
        num_tr_examples_per_class,
        total_examples_per_class - num_tr_examples_per_class, use_cross=FLAGS.cross)
  else:
    metatest_batch = _construct_examples_batch(
        outer_model_config["metatest_batch_size"], "test", num_classes,
        num_tr_examples_per_class,
        15, use_cross=FLAGS.cross)
      
  _, metatest_accuracy, metatest_dacc = _construct_loss_and_accuracy(
      leo, metatest_batch, False)
  _construct_validation_summaries(metavalid_loss, metavalid_accuracy)
  
  break_down_batch = _construct_examples_batch(
      1, "test", num_classes,
      num_tr_examples_per_class,
      15)
  hardness, correct = leo(break_down_batch, False, True, True)
  return (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
          metatest_accuracy, metatrain_dacc, metavalid_dacc, metatest_dacc, hardness, correct)


def run_debug_loop(checkpoint_path):
  outer_model_config = config.get_outer_model_config()
  tf.logging.info("outer_model_config: {}".format(outer_model_config))
  (losses, accuracies, outputs, metatrain_batch, global_step) = construct_debug_graph(outer_model_config)

  num_steps_limit = outer_model_config["num_steps_limit"]
  best_metavalid_accuracy = 0.

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_path,
      save_summaries_steps=FLAGS.checkpoint_steps,
      log_step_count_steps=FLAGS.checkpoint_steps,
      save_checkpoint_steps=FLAGS.checkpoint_steps,
      summary_dir=checkpoint_path) as sess:
    if not FLAGS.evaluation_mode:
      global_step_ev = sess.run(global_step)
      losses_ev, accuracies_ev, outputs_ev, batch = sess.run([losses, accuracies, outputs, metatrain_batch])
      a = 1

def write_output_message(message, file_name=None):
  if file_name is None:
      file_name = "results"
  # output_file = os.path.join(self.args.save_path, "results.txt")
  output_file = os.path.join("outputs", file_name + ".txt")
  with open(output_file, "a") as f:
      f.write(message + "\n")

def run_training_loop(checkpoint_path):
  """Runs the training loop, either saving a checkpoint or evaluating it."""
  outer_model_config = config.get_outer_model_config()
  tf.logging.info("outer_model_config: {}".format(outer_model_config))
  (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
   metatest_accuracy, metatrain_dacc, metavalid_dacc, metatest_dacc, hardness, correct) = construct_graph(outer_model_config)

  num_steps_limit = outer_model_config["num_steps_limit"]
  best_metavalid_accuracy = 0.
  best_metavalid_dacc = 0.
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_path,
      save_summaries_steps=FLAGS.checkpoint_steps,
      log_step_count_steps=FLAGS.checkpoint_steps,
      save_checkpoint_steps=FLAGS.checkpoint_steps,
      summary_dir=checkpoint_path) as sess:
    if not FLAGS.evaluation_mode:
      global_step_ev = sess.run(global_step)
      while global_step_ev < num_steps_limit:
        if global_step_ev % FLAGS.checkpoint_steps == 0:
          # Just after saving checkpoint, calculate accuracy 10 times and save
          # the best checkpoint for early stopping.
          #metavalid_accuracy_ev = utils.evaluate_and_average(
              #sess, metavalid_accuracy, 10)
          metavalid_accuracy_ev, metavalid_dacc_ev = utils.evaluate_and_average_acc_dacc(
              sess, metavalid_accuracy, metavalid_dacc, 10)
          tf.logging.info("Step: {} meta-valid accuracy: {}, dacc: {} best acc: {} best dacc: {}".format(
              global_step_ev, metavalid_accuracy_ev, metavalid_dacc_ev, best_metavalid_accuracy, best_metavalid_dacc))

          if metavalid_accuracy_ev > best_metavalid_accuracy:
            utils.copy_checkpoint(checkpoint_path, global_step_ev,
                                  metavalid_accuracy_ev)
            best_metavalid_accuracy = metavalid_accuracy_ev
          if metavalid_dacc_ev > best_metavalid_dacc:
            best_metavalid_dacc = metavalid_dacc_ev
        _, global_step_ev, metatrain_accuracy_ev = sess.run(
            [train_op, global_step, metatrain_accuracy])
        if global_step_ev % (FLAGS.checkpoint_steps // 2) == 0:
          tf.logging.info("Step: {} meta-train accuracy: {}".format(
              global_step_ev, metatrain_accuracy_ev))
    else:
      if not FLAGS.hacc:
        assert not FLAGS.checkpoint_steps
        num_metatest_estimates = (
            2000 // outer_model_config["metatest_batch_size"])
        # Not changed to dacc yet
        test_accuracy = utils.evaluate_and_average(sess, metatest_accuracy,
                                                  num_metatest_estimates)

        tf.logging.info("Metatest accuracy: %f", test_accuracy)
        with tf.gfile.Open(
            os.path.join(checkpoint_path, "test_accuracy"), "wb") as f:
          pickle.dump(test_accuracy, f)
      else:
        all_hardness = []
        all_correct = []
        for i in range(2000):
          hardness_ev, correct_ev = sess.run([hardness, correct])
          hardness_ev = [hardness_ev[i,:,i] for i in range(5)]
          hardness_ev = np.array(hardness_ev).flatten()
          correct_ev = np.array(correct_ev).flatten()
          all_hardness.append(hardness_ev)
          all_correct.append(correct_ev)
        all_hardness = np.array(all_hardness).flatten()
        all_correct = np.array(all_correct).flatten()
        save_file = {
          "hardness": all_hardness,
          "correct": all_correct
        }
        print(all_correct.sum() / len(all_correct))
        pickle.dump(save_file, open("hacc/" + FLAGS.config, "wb"))



def main(argv):
  del argv  # Unused.
  # print("here")
  ifsl_config = ifsl_configs.__dict__[FLAGS.config]()
  config.load_ifsl_config(ifsl_config)
  run_training_loop(FLAGS.checkpoint_path)
  # run_debug_loop(FLAGS.checkpoint_path)


if __name__ == "__main__":
  # print("here")
  tf.app.run()
