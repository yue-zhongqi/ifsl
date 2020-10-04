# coding=utf8
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
"""A module containing just the configs for the different LEO parts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
import shutil

# python runner.py --config=mini_5_resnet_baseline
# python runner.py --config=mini_5_resnet_baseline --evaluation_mode=True

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to the dataset.")
flags.DEFINE_string(
    "dataset_name", "miniImageNet", "Name of the dataset to "
    "train on, which will be mapped to data.MetaDataset.")
flags.DEFINE_string(
    "embedding_crop", "center", "Type of the cropping, which "
    "will be mapped to data.EmbeddingCrop.")
flags.DEFINE_boolean("train_on_val", False, "Whether to train on the "
                     "validation data.")

flags.DEFINE_integer(
    "inner_unroll_length", 5, "Number of unroll steps in the "
    "inner loop of leo (number of adaptation steps in the "
    "latent space).")
flags.DEFINE_integer(
    "finetuning_unroll_length", 5, "Number of unroll steps "
    "in the loop performing finetuning (number of adaptation "
    "steps in the parameter space).")
flags.DEFINE_integer("num_latents", 64, "The dimensionality of the latent "
                     "space.")
flags.DEFINE_float(
    "inner_lr_init", 1.0, "The initialization value for the "
    "learning rate of the inner loop of leo.")
flags.DEFINE_float(
    "finetuning_lr_init", 0.001, "The initialization value for "
    "learning rate of the finetuning loop.")
flags.DEFINE_float("dropout_rate", 0.5, "Rate of dropout: probability of "
                   "dropping a given unit.")
flags.DEFINE_float(
    "kl_weight", 1e-3, "The weight measuring importance of the "
    "KL in the final loss. β in the paper.")
flags.DEFINE_float(
    "encoder_penalty_weight", 1e-9, "The weight measuring "
    "importance of the encoder penalty in the final loss. γ in "
    "the paper.")
flags.DEFINE_float("l2_penalty_weight", 1e-8, "The weight measuring the "
                   "importance of the l2 regularization in the final loss. λ₁ "
                   "in the paper.")
flags.DEFINE_float("orthogonality_penalty_weight", 1e-3, "The weight measuring "
                   "the importance of the decoder orthogonality regularization "
                   "in the final loss. λ₂ in the paper.")

flags.DEFINE_integer(
    "num_classes", 5, "Number of classes, N in N-way classification.")
flags.DEFINE_integer(
    "num_tr_examples_per_class", 1, "Number of training samples per class, "
    "K in K-shot classification.")
flags.DEFINE_integer(
    "num_val_examples_per_class", 15, "Number of validation samples per class "
    "in a task instance.")
flags.DEFINE_integer("metatrain_batch_size", 12, "Number of problem instances "
                     "in a batch.")
flags.DEFINE_integer("metavalid_batch_size", 200, "Number of meta-validation "
                     "problem instances.")
flags.DEFINE_integer("metatest_batch_size", 200, "Number of meta-testing "
                     "problem instances.")
flags.DEFINE_integer("num_steps_limit", int(1e5), "Number of steps to train "
                     "for.")
flags.DEFINE_float("outer_lr", 1e-4, "Outer (metatraining) loop learning "
                   "rate.")
flags.DEFINE_float(
    "gradient_threshold", 0.1, "The cutoff for the gradient "
    "clipping. Gradients will be clipped to "
    "[-gradient_threshold, gradient_threshold]")
flags.DEFINE_float(
    "gradient_norm_threshold", 0.1, "The cutoff for clipping of "
    "the gradient norm. Gradient norm clipping will be applied "
    "after pointwise clipping (described above).")
flags.DEFINE_string("config", "mini_5_resnet_baseline", "Configuration to use.")
flags.DEFINE_integer("feat_dim", 640, "Feature dimension.")
flags.DEFINE_boolean("deconfound", False, "Whether to deconfound.")
flags.DEFINE_boolean("use_test", False, "Whether to use test.")
flags.DEFINE_boolean("retrain", False, "Whether to discard saved checkpoints and retrain.")
flags.DEFINE_integer("num_pretrain_classes", 64, "Number of classes in pre-train dataset")
flags.DEFINE_string("pretrain_mean_filename", "miniImagenet_simpleshot_ResNet10_mean.npy", "Pretrain mean npy file name")
flags.DEFINE_integer("n_splits", 4, "Number of splits")
flags.DEFINE_boolean("is_cosine_feature", True, "Is it cosine feature")
flags.DEFINE_string("fusion", "concat", "How to fuse feature.")
flags.DEFINE_string("classifier", "single", "Classifier design")
flags.DEFINE_integer("pretrain_num_classes", 64, "Number of classes in pretrain dataset")
flags.DEFINE_string("logit_fusion", "product", "When using bi classifier, the logit fusion function to use")
flags.DEFINE_boolean("use_x_only", False, "Only using X feature")
flags.DEFINE_string("preprocess_before_split", "none", "Preprocessing before split")
flags.DEFINE_string("preprocess_after_split", "none", "Preprocessing after split")
flags.DEFINE_boolean("normalize_before_center", True, "Normalizing feature before centering operation")
flags.DEFINE_boolean("normalize_d", False, "Normalizing d features")
flags.DEFINE_boolean("normalize_ed", False, "Normalizing ed features")
flags.DEFINE_boolean("hacc", False, "Turn on saving hacc for evaluation")
flags.DEFINE_boolean("cross", False, "Evaluating on cross")


def get_data_config():
  config = {}
  config["data_path"] = FLAGS.data_path
  config["dataset_name"] = FLAGS.dataset_name
  config["embedding_crop"] = FLAGS.embedding_crop
  config["train_on_val"] = FLAGS.train_on_val
  config["total_examples_per_class"] = 600
  return config


def get_inner_model_config():
  """Returns the config used to initialize LEO model."""
  config = {}
  config["inner_unroll_length"] = FLAGS.inner_unroll_length
  config["finetuning_unroll_length"] = FLAGS.finetuning_unroll_length
  config["num_latents"] = FLAGS.num_latents
  config["inner_lr_init"] = FLAGS.inner_lr_init
  config["finetuning_lr_init"] = FLAGS.finetuning_lr_init
  config["dropout_rate"] = FLAGS.dropout_rate
  config["kl_weight"] = FLAGS.kl_weight
  config["encoder_penalty_weight"] = FLAGS.encoder_penalty_weight
  config["l2_penalty_weight"] = FLAGS.l2_penalty_weight
  config["orthogonality_penalty_weight"] = FLAGS.orthogonality_penalty_weight
  config["feat_dim"] = FLAGS.feat_dim
  config["pretrain_mean_filename"] = FLAGS.pretrain_mean_filename
  return config


def get_outer_model_config():
  """Returns the outer config file for N-way K-shot classification tasks."""
  config = {}
  config["num_classes"] = FLAGS.num_classes
  config["num_tr_examples_per_class"] = FLAGS.num_tr_examples_per_class
  config["num_val_examples_per_class"] = FLAGS.num_val_examples_per_class
  config["metatrain_batch_size"] = FLAGS.metatrain_batch_size
  config["metavalid_batch_size"] = FLAGS.metavalid_batch_size
  config["metatest_batch_size"] = FLAGS.metatest_batch_size
  config["num_steps_limit"] = FLAGS.num_steps_limit
  config["outer_lr"] = FLAGS.outer_lr
  config["gradient_threshold"] = FLAGS.gradient_threshold
  config["gradient_norm_threshold"] = FLAGS.gradient_norm_threshold
  return config


def load_ifsl_config(config):
    # dataset_name, number of pretrain classes
    if config.dataset == "miniImagenet":
        FLAGS.dataset_name = "miniImageNet"
        FLAGS.num_pretrain_classes = 64
    elif config.dataset == "tiered":
        FLAGS.dataset_name = "tieredImageNet"
        FLAGS.num_pretrain_classes = 351
    # checkpoint path
    FLAGS.checkpoint_path = "/data2/yuezhongqi/Model/leo/ifsl/" + config.dataset + "_" + config.model + "_" + \
                            str(config.shot) + "_" + config.meta_label
    # data path
    if config.model == "ResNet10":
        model_abbr = "resnet"
    elif config.model == "wideres":
        model_abbr = "wrn"
    FLAGS.data_path = "/data2/yuezhongqi/Model/leo/" + model_abbr + "_noaug_embeddings"
    # pretrain mean filename
    FLAGS.pretrain_mean_filename = config.dataset + "_" + config.method + "_" + config.model + "_mean.npy"
    # shot
    FLAGS.num_tr_examples_per_class = config.shot
    # test iter: Default is 2000, which is desired
    # deconfound
    FLAGS.deconfound = config.deconfound
    # feature dimension
    if config.model == "ResNet10":
        FLAGS.feat_dim = 512
    elif config.model == "wideres":
        FLAGS.feat_dim = 640
    # evaluation mode
    if FLAGS.evaluation_mode:
        FLAGS.checkpoint_steps = 0
        FLAGS.retrain = False
    # hyperparameter settings
    if config.shot == 5 and config.dataset == "miniImagenet":
        FLAGS.outer_lr = 4.1024e-4
        FLAGS.l2_penalty_weight = 8.54e-9
        FLAGS.orthogonality_penalty_weight = 1.523998e-3
        FLAGS.dropout_rate = 0.300299
        FLAGS.kl_weight = 0.466387
        FLAGS.encoder_penalty_weight = 2.661608e-7
    elif config.shot == 1 and config.dataset == "miniImagenet":
        FLAGS.outer_lr = 2.739071e-4
        FLAGS.l2_penalty_weight = 3.623413e-10
        FLAGS.orthogonality_penalty_weight = 0.188103
        FLAGS.dropout_rate = 0.307651
        FLAGS.kl_weight = 0.756143
        FLAGS.encoder_penalty_weight = 5.756821e-6
    elif config.shot == 1 and config.dataset == "tiered":
        FLAGS.outer_lr = 8.659053e-4
        FLAGS.l2_penalty_weight = 4.148858e-10
        FLAGS.orthogonality_penalty_weight = 5.451078e-3
        FLAGS.dropout_rate = 0.475126
        FLAGS.kl_weight = 2.034189e-3
        FLAGS.encoder_penalty_weight = 8.302962e-5
    elif config.shot == 5 and config.dataset == "tiered":
        FLAGS.outer_lr = 6.110314e-4
        FLAGS.l2_penalty_weight = 1.690399e-10
        FLAGS.orthogonality_penalty_weight = 2.481216e-2
        FLAGS.dropout_rate = 0.415158
        FLAGS.kl_weight = 1.622811
        FLAGS.encoder_penalty_weight = 2.672450e-5
    # retrain
    if FLAGS.retrain:
        if os.path.isdir(FLAGS.checkpoint_path):
            shutil.rmtree(FLAGS.checkpoint_path)
    # IFSL parameters
    if config.deconfound:
        FLAGS.n_splits = config.n_splits
        FLAGS.is_cosine_feature = config.is_cosine_feature
        FLAGS.fusion = config.fusion
        FLAGS.classifier = config.classifier
        FLAGS.pretrain_num_classes = config.num_classes
        FLAGS.logit_fusion = config.logit_fusion
        FLAGS.use_x_only = config.use_x_only
        FLAGS.preprocess_before_split = config.preprocess_before_split
        FLAGS.preprocess_after_split = config.preprocess_after_split
        FLAGS.normalize_before_center = config.normalize_before_center
        FLAGS.normalize_d = config.normalize_d
        FLAGS.normalize_ed = config.normalize_ed
    # Overwrite default hyperparameter settings
    if hasattr(config, "outer_lr"):
        FLAGS.outer_lr = config.outer_lr