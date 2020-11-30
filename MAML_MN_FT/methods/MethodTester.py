import os
import data.feature_loader as feat_loader
import numpy as np
import random
import torch
import configs
from io_utils import print_with_carriage_return
from io_utils import end_carriage_return_print
from io_utils import model_dict
from io_utils import append_to_file
from io_utils import get_result_file

from models.SimpleShotResNet import resnet10
from models.SimpleShotWideResNet import wideres
from models.FeatWRN import FEATWRN

from methods.PretrainedModel import PretrainedModel
from data.datamgr import SetDataManager


class MethodTester():
    def __init__(self, params):
        self.params = params
        # self.initialize(params)
        self.params.iter_num = 600
        # Experiment
        self.early_stop_criteria = []

    def get_backbone(self):
        if self.params.method == "S2M2_R":
            backbone = wrn28_10
        elif self.params.method == "baseline" or self.params.method == "baseline++":
            backbone = model_dict[self.params.model]
        elif self.params.method == "simpleshot":
            backbone = resnet10
        elif self.params.method == "simpleshotwide":
            backbone = wideres
        elif self.params.method == "feat":
            backbone = FEATWRN
        elif self.params.method == "sib":
            backbone = SIBWRN
        return backbone

    def baseline_s2m2_initialize(self, params, provide_original_image):
        if params.dataset in ['omniglot', 'cross_char']:
            assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
            params.model = 'Conv4S'
        
        dataset = params.dataset
        if params.dataset == "cross":
            dataset = "miniImagenet"
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, dataset, params.model, params.method)

        if params.train_aug:
            checkpoint_dir += '_aug'
        if params.method not in ['baseline', 'baseline++', 'S2M2_R']:
            checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

        self.checkpoint_dir = checkpoint_dir

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        # defaut split = novel, but you can also test base or val classes
        novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split_str + ".hdf5")
        
        if params.dataset == "cross":
            novel_file = novel_file.replace("miniImagenet", "cross")

        file_path = "img_paths_%s_%s_%s.npy" % (params.dataset, params.model, params.method)
        if provide_original_image:
            self.cl_data_file, self.path_data_file = feat_loader.init_loader(novel_file, provide_original_image, file_path)
        else:
            self.cl_data_file = feat_loader.init_loader(novel_file, provide_original_image, file_path)
        self.image_size = 224
        if params.method == "S2M2_R":
            self.image_size = 80

    def simpleshot_initialize(self, params, provide_original_image):
        novel_file = '%s/features/%s/%s/novel.hdf5' % (configs.simple_shot_dir, params.dataset, params.model)
        # novel_file = "/model/1154027137/ifsl_features/features/" + params.method + "_" + params.dataset + "_novel.hdf5"
        image_file_path = "img_paths_%s_%s_%s.npy" % (params.dataset, params.model, params.method)
        if provide_original_image:
            self.cl_data_file, self.path_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        else:
            self.cl_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        self.image_size = 84

    def feat_initialize(self, params, provide_original_image):
        novel_file = '%s/features/%s/%s/novel.hdf5' % (configs.feat_dir, params.dataset, params.model)
        # novel_file = "/model/1154027137/ifsl_features/features/" + params.method + "_" + params.dataset + "_novel.hdf5"
        image_file_path = "img_paths_%s_%s_%s.npy" % (params.dataset, params.model, params.method)
        if provide_original_image:
            self.cl_data_file, self.path_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        else:
            self.cl_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        self.image_size = 84

    def sib_initialize(self, params, provide_original_image):
        novel_file = '%s/features/%s/%s/novel.hdf5' % (configs.sib_dir, params.dataset, params.model)
        image_file_path = "img_paths_%s_%s_%s.npy" % (params.dataset, params.model, params.method)
        if provide_original_image:
            self.cl_data_file, self.path_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        else:
            self.cl_data_file = feat_loader.init_loader(novel_file, provide_original_image, image_file_path)
        self.image_size = 80

    def initialize(self, params, provide_original_image=False):
        self.params = params
        if params.method == "baseline" or params.method == "baseline++" or params.method == "S2M2_R":
            self.baseline_s2m2_initialize(params, provide_original_image)
        elif params.method in ["simpleshot", "simpleshotwide"]:
            self.simpleshot_initialize(params, provide_original_image)
        elif params.method in ["feat"]:
            self.feat_initialize(params, provide_original_image)
        elif params.method in ["sib"]:
            self.sib_initialize(params, provide_original_image)
        self.params.n_query = 15
        # self.params.n_query = 8
        self.params.adaptation = True
        self.few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=self.params.n_query)
    
    def get_task(self, all_from_same_class=False, provide_original_image=False):
        # return n_way * (n_shot + n_query) images. arranged as 0,0,0,...1,1,1...
        class_list = self.cl_data_file.keys()

        select_class = random.sample(class_list, self.params.test_n_way)
        if all_from_same_class:
            select_class = np.full(self.params.test_n_way, random.sample(class_list, 1)[0])
        z_all = []
        image_path_all = []
        for cl in select_class:
            img_feat = self.cl_data_file[cl]
            if provide_original_image:
                img_path = self.path_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            # stack each batch
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(self.params.n_shot + self.params.n_query)])
            if provide_original_image:
                image_path_all.append([str(img_path[perm_ids[i]]) for i in range(self.params.n_shot + self.params.n_query)])
        z_all = torch.from_numpy(np.array(z_all) )
        if provide_original_image:
            self.image_path_all = image_path_all
        return z_all

    def get_task_special(self, sampling="sim"):
        n_way = self.params.test_n_way
        n_support = self.params.n_shot
        n_query = self.params.n_query
        pretrain = self.pretrain
        num_classes = 64
        min_p = 0.9

        task = np.zeros((n_way, n_support + n_query, pretrain.feat_dim))
        if not hasattr(self, "features"):
            features, labels = pretrain.get_pretrain_dataset("novel")
            self.features = features
            self.labels = labels
            preds = []
            for feature in features:
                batch_feature = torch.from_numpy(feature).cuda().unsqueeze(0)
                prob = pretrain.classify(batch_feature).squeeze(0).cpu().numpy()[:num_classes]
                preds.append(prob)
            preds = np.asarray(preds)
            self.preds = preds
        else:
            features = self.features
            labels = self.labels
            preds = self.preds
        
        classes = np.unique(labels)
        samples = {}

        for cl in classes:
            cl_idx = (labels == cl)
            cl_features = features[cl_idx]
            cl_preds = preds[cl_idx]
            cl_features_selected = []
            preds_selected = []
            for i in range(len(cl_features)):
                if cl_preds[i].max() >= min_p:
                    cl_features_selected.append(cl_features[i])
                    preds_selected.append(cl_preds[i])
            # check for valid classes
            cl_valid_classes = []
            for i in range(num_classes):
                n_valid_support = 0
                for j in range(len(cl_features_selected)):
                    if np.argmax(preds_selected[j]) == i:
                        n_valid_support += 1
                if n_valid_support > n_support:
                    cl_valid_classes.append(i)
            samples[cl] = {}
            samples[cl]["features"] = np.asarray(cl_features_selected)
            samples[cl]["preds"] = np.asarray(preds_selected)
            samples[cl]["valid_cls"] = np.asarray(cl_valid_classes)
        
        episode_cl = np.random.choice(classes, n_way)
        for j in range(n_way):
            # 2. sample a prototype for each class
            success = False
            current_cl = episode_cl[j]
            current_samples = samples[current_cl]
            prototype_base_class = -1
            while not success:
                idx = np.random.randint(0, len(current_samples["features"]))
                if np.argmax(current_samples["preds"][idx]) in current_samples["valid_cls"]:
                    success = True
                    prototype_base_class = np.argmax(current_samples["preds"][idx])
            support_valid_idx = []
            query_valid_idx = []
            query_base_class = current_samples["valid_cls"][-2]
            if query_base_class == prototype_base_class:
                query_base_class = current_samples["valid_cls"][-1]
            for k in range(len(current_samples["features"])):
                if np.argmax(current_samples["preds"][k]) == prototype_base_class:
                    support_valid_idx.append(k)
                #elif np.argmax(current_samples["preds"][k]) == query_base_class:
                else:
                    query_valid_idx.append(k)
            support_valid_idx = np.asarray(support_valid_idx)
            query_valid_idx = np.asarray(query_valid_idx)
            support_idx = np.random.choice(support_valid_idx, n_support)
            query_idx = np.random.choice(query_valid_idx, n_query)
            if sampling == "sim":
                t = np.random.choice(support_valid_idx, n_support + n_query)
                support_idx = t[:n_support]
                query_idx = t[n_support:]
            # 3. fill support set with images similar to prototype in pred
            for k in range(n_support):
                task[j][k] = current_samples["features"][support_idx[k]]
            # 4. fill query set with images different from prototype
            for k in range(n_query):
                task[j][n_support + k] = current_samples["features"][query_idx[k]]
        task = torch.from_numpy(task).cuda().float()
        return task

    def set_experiment_config(self, config):
        self.experiment_config = config

    def set_conditional_config(self, config_func):
        self.conditional_config_func = config_func

    def set_experiment_method(self, method):
        self.experiment_method = method

    def add_early_stop_criteria(self, iter, acc):
        '''
        Experiment will terminate early when test accuracy < acc% at test iteration >= iter
        An experiment can have multiple early stop criteria
        '''
        self.early_stop_criteria.append({"iter": iter, "acc": acc})

    def _should_reinitialize(self, config):
        return ("dataset" in config) or ("model" in config) or ("method" in config) or ("n_shot" in config) or ("pretrain" in config)
    
    def start_experiment(self, method, config, test_name="No name", conditional_config_func=None,
                         show_current_accuracy=True, use_fixed_configs=False, save_result=True,
                         provide_original_image=False, continue_from=-1, require_pretrain=False,
                         use_meta_module="none", sampling="average", metric="normal"):
        if not use_fixed_configs:
            # configs from config dict
            config_list = self._get_config_list(config)
            
            # add configs from conditional config function
            if (conditional_config_func):
                combined_config_list = []
                for config_dict in config_list:
                    conditional_config = conditional_config_func(config_dict)
                    cond_config_list = self._get_config_list(conditional_config)
                    for cond_config in cond_config_list:
                        new_config_dict = dict(config_dict)
                        new_config_dict.update(cond_config)
                        combined_config_list.append(new_config_dict)
            else:
                combined_config_list = config_list
        else:
            combined_config_list = config
        
        counter = 0

        if "method" in config:
            method_name = config["method"][0]
        else:
            method_name = self.params.method
        save_file_path = get_result_file(test_name=test_name, method_name=method_name)

        if continue_from > 0:
            combined_config_list = combined_config_list[continue_from:]
            counter = 0
            
        for config in combined_config_list:
            counter += 1
            print("Running config %d/%d" % (counter, len(combined_config_list)))
            test_case = self._generate_test_case_name(config)
            if use_meta_module != "none":
                test_case += " meta_module:%s" % (use_meta_module)
            print(test_case)

            # In case model or dataset or method is in config, re-init is required
            if self._should_reinitialize(config) or provide_original_image:
                if "dataset" in config:
                    self.params.dataset = config["dataset"]
                    config.pop("dataset", None)
                if "model" in config:
                    self.params.model = config["model"]
                    config.pop("model", None)
                if "method" in config:
                    self.params.method = config["method"]
                    config.pop("method", None)
                if "n_shot" in config:
                    self.params.n_shot = config["n_shot"]
                    config.pop("n_shot", None)
                if "pretrain" in config or require_pretrain:
                    # re-construct pretrained model due to dataset and model change
                    config["pretrain"] = PretrainedModel(self.params)
                    self.pretrain = config["pretrain"]
            # !! Change to initialize every time
            self.initialize(self.params, provide_original_image)

            if save_result:
                append_to_file(save_file_path, test_case)

            backbone = self.get_backbone()
            
            model = method(backbone, **self.few_shot_params, **config)
            model.eval()
            print("Using metric: %s" % (metric))
            result = self._run_model(model, show_current_accuracy, provide_original_image, sampling, metric=metric)
            print(result)
            
            if save_result:
                append_to_file(save_file_path, result)
            # Release memory
            config["pretrain"] = None
            config["meta_module"] = None

    def _get_config_list(self, config_dict):
        if (len(config_dict) == 0):
            return []

        config_list = [{}]
        for key in config_dict:
            tmp_config_list = []
            for config in config_list:
                for val in config_dict[key]:
                    new_config = dict(config)
                    new_config[key] = val
                    tmp_config_list.append(new_config)
            config_list = tmp_config_list
        return config_list
                    
    def _increment_config_counter(self, config_dict, config_key_counter):
        for key in config_dict:
            key_current_counter = config_key_counter[key]
            if key_current_counter < len(config_dict[key]) - 1:
                config_key_counter[key] += 1
                return True
        return False

    def _generate_test_case_name(self, config):
        test_case_name = ""
        for key in config:
            if key != "pretrain":
                test_case_name += "%s:%s " % (key, str(config[key]))
        return test_case_name

    def _run_model(self, model, show_current_accuracy=True, provide_original_image=False, sampling="average", metric="normal"):
        acc_all = []
        did_early_stop = False
        for i in range(self.params.iter_num):
            acc, diff_score = self._feature_evaluation(model, provide_original_image, sampling=sampling, metric=metric)
            acc_all.append(acc)
            #diff_all.append(diff_score)
            #d_acc_all.append(acc * diff_score)
            mean_acc = np.mean(acc_all)
            #mean_d_acc = np.sum(d_acc_all) / np.sum(diff_all)
            if show_current_accuracy:
                acc_report = "%d/%d: Mean accuracy: %.4f" % (i, self.params.iter_num, mean_acc)
                # acc_report = "%d/%d: Mean accuracy: %.4f" % (i, self.params.iter_num, mean_d_acc)
                print_with_carriage_return(acc_report)
            if self._should_early_stop(i, mean_acc):
                did_early_stop = True
                break

        if show_current_accuracy:
            end_carriage_return_print()

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if did_early_stop:
            result = "EARLY STOP %d Test Acc = %4.2f%% +- %4.2f%%" %\
                (i, acc_mean, 1.96 * acc_std / np.sqrt(self.params.iter_num))
        else:
            result = '%d Test Acc = %4.2f%% +- %4.2f%%' %\
                (i + 1, acc_mean, 1.96 * acc_std / np.sqrt(self.params.iter_num))
        return result

    def _feature_evaluation(self, model, provide_original_image, sampling="average", metric="normal"):
        if sampling == "average":
            z_all = self.get_task(provide_original_image=provide_original_image)
        else:
            z_all = self.get_task_special(sampling)
            
        if provide_original_image:
            image_paths = self.image_path_all
        scores = None
        model.n_query = self.params.n_query
        if self.params.adaptation:
            if provide_original_image:
                scores = model.set_forward_adaptation(z_all, image_paths=image_paths, is_feature=True)
            else:
                scores = model.set_forward_adaptation(z_all, is_feature=True)
        else:
            scores = model.set_forward(z_all, is_feature=True)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.params.test_n_way), self.params.n_query)
        # precision, recall, profile = calc_recall_precision(y, pred)
        # print(diff_score)
        if metric == "dacc":
            #diff_score = np.array(self._evaluate_hardness(z_all))
            diff_score = np.array(self._evaluate_hardness_logodd(z_all))
            acc = 100 * ((pred == y) * diff_score).sum() / diff_score.sum()
        else:
            acc = np.mean(pred == y) * 100
        return acc, 0

    def cosine_similarity(self, a, b):
        return (a * b).sum() / torch.norm(a, p=2) / torch.norm(b, p=2)

    def _evaluate_hardness(self, z_all):
        z_all = z_all.cuda()
        pretrain = self.pretrain
        num_classes = self.pretrain.num_classes
        probs = torch.zeros(z_all.shape[0], z_all.shape[1], num_classes).cuda()
        for i in range(self.params.train_n_way):
            #for j in range(self.params.n_shot + self.params.n_query):
                #probs[i][j] = pretrain.classify(z_all[i][j])[:]
            probs[i] = pretrain.classify(z_all[i])[:, : num_classes]

        total_diff_scores = []
        for i in range(self.params.train_n_way):
            for j in range(self.params.n_query):
                current_query_prob = probs[i][self.params.n_shot + j]
                current_query_diff_score = 0
                for k in range(self.params.n_shot):
                    current_support_prob = probs[i][k]
                    similarity = self.cosine_similarity(current_query_prob, current_support_prob)
                    current_query_diff_score += (1 - similarity)
                '''
                for k in range(self.params.train_n_way):
                    if k != i:
                        for l in range(self.params.n_shot):
                            current_support_prob = probs[k][l]
                            similarity = self.cosine_similarity(current_query_prob, current_support_prob)
                            current_query_diff_score += similarity
                current_query_diff_score /= (self.params.n_shot * self.params.train_n_way)
                '''
                current_query_diff_score /= self.params.n_shot
                total_diff_scores.append(current_query_diff_score.cpu().numpy())
        # diff_score = total_diff_scores / self.params.train_n_way / self.params.n_query / self.params.n_shot
        return total_diff_scores

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def _evaluate_hardness_logodd(self, z_all):
        z_all = z_all.cuda()
        pretrain = self.pretrain
        num_classes = self.pretrain.num_classes
        n_shot = self.params.n_shot
        n_way = self.params.train_n_way
        n_query = self.params.n_query
        probs = torch.zeros(z_all.shape[0], z_all.shape[1], num_classes).cuda()
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=1)
        for i in range(self.params.train_n_way):
            probs[i] = pretrain.classify(z_all[i], normalize_prob=False)[:, : num_classes]
            probs[i] = relu(probs[i])
        # 1. Obtain 5*64 weights
        w = probs[:, :n_shot, :].mean(dim=1)
        w = self.normalize(w)
        # 2. Obtain query in 75 * 64 format
        query = probs[:, n_shot:, :]  # 5*15*64
        query = query.contiguous().view(n_way * n_query, -1)  # 75 * 64
        query = self.normalize(query)
        y_query = np.repeat(range(n_way), n_query)
        w = w.unsqueeze(0).expand(query.shape[0], -1, -1)
        query = query.unsqueeze(1).expand(-1, n_way, -1)
        logits = (w * query).sum(dim=2)  # 75 * 5
        query_probs = softmax(logits)
        hardness = []
        # correct = (query_probs.argmax(dim=1).cpu().numpy() == y_query).sum()
        # self.tmp_correct = correct
        for i in range(query_probs.shape[0]):
            p = query_probs[i][y_query[i]]
            log_odd = torch.log((1 - p) / p)
            hardness.append(log_odd.cpu().numpy())
        hardness = np.array(hardness)
        if hardness.min() < 0:
            hardness -= hardness.min()
        return hardness
    
    def _should_early_stop(self, iter, acc):
        for criteria in self.early_stop_criteria:
            if criteria["iter"] <= iter:
                if criteria["acc"] >= acc:
                    return True
        return False

    def meta_train(self, config, method, descriptor_str, debug=True, use_test=False, require_pretrain=False, metric="acc"):
        config["meta_training"] = True
        params = self.params
        params.save_freq = 10
        params.n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
        params.dataset = config["dataset"]
        params.model = config["model"]
        params.method = config["method"]
        params.n_shot = config["n_shot"]
        train_episodes = config["train_episodes"]
        val_episodes = config["val_episodes"]
        end_epoch = config["end_epoch"]
        if "weight_decay" in config:
            weight_decay = config["weight_decay"]
        else:
            weight_decay = 0

        result_dir = "results/meta/%s" % (params.dataset)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_file = os.path.join(result_dir, "%s_%s_%s.txt" % (params.method, params.model, descriptor_str))

        self.few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=self.params.n_query)
        params.checkpoint_dir = '%s/checkpoints/%s/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.method, params.model, descriptor_str, params.n_shot)
        params.stop_epoch = 100
        self.initialize(params, False)
        image_size = self.image_size
        pretrain = PretrainedModel(self.params)

        if use_test:
            file_name = "novel.json"
        else:
            file_name = "val.json"
        if params.dataset == 'cross':
            base_file = configs.data_dir['miniImagenet'] + 'base.json'
            val_file = configs.data_dir['CUB'] + file_name
        else:
            base_file = configs.data_dir[params.dataset] + 'base.json'
            val_file = configs.data_dir[params.dataset] + file_name
        
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way))

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params, n_eposide=train_episodes)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug, debug=debug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params, n_eposide=val_episodes)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False, debug=debug)

        backbone = self.get_backbone()
        if "params" in config:
            model_params = config["params"]
        else:
            model_params = {}
        if require_pretrain:
            model_params["pretrain"] = pretrain

        model = method(backbone, **train_few_shot_params, **model_params)
        model = model.cuda()

        # Freeze backbone
        model.feature = None
        if not require_pretrain:
            model.pretrain = pretrain
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

        max_acc = 0
        for epoch in range(0, end_epoch):
            model.epoch = epoch
            model.train()
            model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
            model.eval()

            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.test_loop(val_loader, metric=metric)
            message = "Epoch: %d, Validation accuracy: %.3f, Best validation accuracy: %.3f" % (epoch, acc, max_acc)
            print(message)
            append_to_file(result_file, message)

            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == params.stop_epoch - 1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        self.meta_test(config, method, descriptor_str, debug, require_pretrain)

    def meta_test(self, config, method, descriptor_str, debug=True, require_pretrain=False, metric="acc"):
        config["meta_training"] = True
        params = self.params
        params.save_freq = 50
        params.n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
        params.dataset = config["dataset"]
        params.model = config["model"]
        params.method = config["method"]
        params.n_shot = config["n_shot"]

        self.few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=self.params.n_query)
        params.checkpoint_dir = '%s/checkpoints/%s/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.method, params.model, descriptor_str, params.n_shot)
        params.stop_epoch = 2000
        self.initialize(params, False)
        image_size = self.image_size
        pretrain = PretrainedModel(self.params)

        if params.dataset == 'cross':
            test_file = configs.data_dir['CUB'] + 'novel.json'
        else:
            test_file = configs.data_dir[params.dataset] + 'novel.json'

        n_query = 15

        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        datamgr = SetDataManager(image_size, n_query=n_query, **few_shot_params, n_eposide=params.stop_epoch)
        loader = datamgr.get_data_loader(test_file, aug=False, debug=debug)

        if "params" in config:
            model_params = config["params"]
        else:
            model_params = {}
        if require_pretrain:
            model_params["pretrain"] = pretrain

        backbone = self.get_backbone()
        model = method(backbone, **few_shot_params, **model_params)
        model = model.cuda()
        model_file = os.path.join(params.checkpoint_dir, "best_model.tar")
        # Load model
        state_dict = model.state_dict()
        saved_states = torch.load(model_file)["state"]
        state_dict.update(saved_states)
        model.load_state_dict(state_dict)
        # Freeze backbone
        model.feature = None
        model.pretrain = pretrain
        
        model.eval()
        acc = model.test_loop(loader, metric=metric)
        print(acc)
