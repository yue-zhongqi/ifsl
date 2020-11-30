import numpy as np
import torch
import torch.nn as nn
from io_utils import model_dict, get_best_file, get_assigned_file, print_with_carriage_return, end_carriage_return_print
import configs
from data.datamgr import SimpleDataManager
from torch.autograd import Variable
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import torchfile
from models.SimpleShotResNet import resnet10
from models.SimpleShotWideResNet import wideres
from models.FeatWRN import FEATWRN

class PretrainedModel():
    def __init__(self, params):
        self.params = params
        self.method = params.method
        self.dataset = params.dataset
        self.model_name = params.model
        if self.dataset == "cross":
            self.base_dataset = "miniImagenet"
        else:
            self.base_dataset = self.dataset
        if self.method in ['baseline', 'baseline++', 'S2M2_R']:
            self.baseline_s2m2_init(params)
        elif self.method in ["simpleshot", "simpleshotwide"]:
            self.simpleshot_init(params)
        elif self.method in ["feat"]:
            self.feat_init(params)
        elif self.method in ["sib"]:
            self.sib_init(params)
        # self.train_d_specific_classifiers(8)
        # self.train_d_specific_classifiers(16)
        # self.load_d_specific_classifiers(10)
        # self.test_d_specific_classifiers(10)
        
    def baseline_s2m2_init(self, params):
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, self.base_dataset, params.model, params.method)

        if params.train_aug:
            checkpoint_dir += '_aug'
        if params.method not in ['baseline', 'baseline++', 'S2M2_R']:
            checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)
        self.num_classes = 200
        if params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], self.num_classes, loss_type='dist')
        elif params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], self.num_classes, loss_type='softmax')
        elif params.method == 'S2M2_R':
            model = wrn28_10(200)
        model = model.cuda()

        if params.method == 'S2M2_R':
            tmp = torch.load(modelfile)
            state = tmp['state']
            # Update feature extractor
            model_dict_load = model.state_dict()
            model_dict_load.update(state)
            model.load_state_dict(model_dict_load)
            # Update classifier
            cls_dict = model.linear.state_dict()
            cls_dict.update(tmp['classifier'])
            model.linear.load_state_dict(cls_dict)
        else:
            tmp = torch.load(modelfile)
            state = tmp['state']
            model.load_state_dict(state)
        model.eval()
        self.model = model
        if self.method == 'S2M2_R':
            self.feat_dim = 640
            self.image_size = 80
            self.batch_size = 16
        else:
            self.feat_dim = 512
            self.image_size = 224
            self.batch_size = 64

    def simpleshot_init(self, params):
        if params.dataset == "tiered":
            self.num_classes = 351
        else:
            self.num_classes = 64
        model_name = params.model.lower()

        if model_name == "resnet10":
            model_abbr = "resnet"
        elif model_name == "wideres":
            model_abbr = "wrn"
        if params.dataset == "cross" or params.dataset == "miniImagenet":
            # model_dir = "/model/1154027137/ifsl_mini_pretrain/ifsl_mini/ss_" + model_abbr + "_mini.tar"
            model_dir = os.path.join(configs.simple_shot_dir, "miniImagenet", model_name, "model_best.pth.tar")
        elif params.dataset == "tiered":
            # model_dir = "/model/1154027137/ifsl_tiered_pretrain/ifsl_tiered/ss_" + model_abbr + "_tiered.tar"
            model_dir = os.path.join(configs.simple_shot_dir, "tiered", model_name, "model_best.pth.tar")

        def remove_module_from_param_name(params_name_str):
            split_str = params_name_str.split(".")[1:]
            params_name_str = ".".join(split_str)
            return params_name_str

        if model_name == "resnet10":
            model = resnet10(num_classes=self.num_classes, remove_linear=False)
        elif model_name == "wideres":
            model = wideres(num_classes=self.num_classes, remove_linear=False)
        # model = simple_shot_models.__dict__[model_name](num_classes=self.num_classes, remove_linear=False)
        model = model.cuda()
        checkpoint = torch.load(model_dir)
        model_dict = model.state_dict()
        model_params = checkpoint['state_dict']
        model_params = {remove_module_from_param_name(k): v for k, v in model_params.items()}
        model_params = {k: v for k, v in model_params.items() if k in model_dict}
        model_dict.update(model_params)
        model.load_state_dict(model_dict)
        model.eval()
        self.model = model
        self.image_size = 84
        if self.model_name == "wideres":
            self.batch_size = 128
            self.feat_dim = 640
        else:
            self.batch_size = 128
            self.feat_dim = 512

    def feat_init(self, params):
        model_name = params.model.lower()      # wrn
        self.num_classes = 64
        if params.dataset == "cross" or params.dataset == "miniImagenet":
            model_dir = os.path.join(configs.feat_dir, "miniImagenet", model_name + "_pre.pth")
            # model_dir = "/model/1154027137/ifsl_mini_pretrain/ifsl_mini/ft_" + "wrn" + "_mini.tar"
        model = FEATWRN(num_classes=self.num_classes)
        model = model.cuda()

        model_dict = model.state_dict()
        checkpoint = torch.load(model_dir)
        model_dict.update(checkpoint['params'])
        model.load_state_dict(model_dict)
        model.eval()
        self.model = model
        self.image_size = 84
        self.batch_size = 64
        self.feat_dim = 640

    def sib_init(self, params):
        model_name = params.model.lower()      # wrn
        self.num_classes = 64
        if params.dataset == "cross" or params.dataset == "miniImagenet":
            model_dir = os.path.join(configs.sib_dir, "miniImagenet", model_name + "_best.pth")
            clf_dir = os.path.join(configs.sib_dir, "miniImagenet", model_name + "_best_clf.pth")
        model = SIBWRN(num_classes=self.num_classes)
        model = model.cuda()

        # Encoder Init
        model_dict = model.encoder.state_dict()
        checkpoint = torch.load(model_dir)
        model_dict.update(checkpoint)
        model.encoder.load_state_dict(model_dict)

        # Classifier Init
        clf_dict = model.classifier.state_dict()
        clf_dict.update(torch.load(clf_dir))
        model.classifier.load_state_dict(clf_dict)
        model.eval()
        self.model = model
        self.image_size = 80
        self.batch_size = 64
        self.feat_dim = 640

    def get_features(self, x):
        with torch.no_grad():
            if self.method == 'S2M2_R':
                features, _ = self.model(x)
                features = features
            elif self.method in ["simpleshot", "simpleshotwide"]:
                features, _ = self.model(x, feature=True)
                features = features
            elif self.method in ["feat", "sib"]:
                features = self.model.forward_feature(x)
            else:
                features = self.model.feature(x)
        return features

    def classify(self, x, normalize_prob=True):
        softmax = torch.nn.Softmax(dim=1)
        if self.method == 'S2M2_R':
            logit = self.model.linear(x)
        elif self.method in ["simpleshot"]:
            logit = self.model.fc(x)
        elif self.method == "simpleshotwide":
            logit = self.model.linear(x)
        elif self.method in ["feat"]:
            logit = self.model.FC(x)
        elif self.method in ["sib"]:
            logit = self.model.classifier(x)
        else:
            logit = self.model.classifier.forward(x)
        if normalize_prob:
            prob = softmax(logit)
        else:
            prob = logit
        prob_no_grad = prob.detach()
        return prob_no_grad

    def load_d_specific_classifiers(self, n_clf):
        in_dim = int(self.feat_dim / n_clf)
        out_dim = self.num_classes
        self.clfs = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(n_clf)])
        self.clfs = self.clfs.cuda()
        out_dir = "pretrain/clfs/%s_%s_%s_%d" % (self.method, self.model_name, self.base_dataset, n_clf)
        outfile = os.path.join(out_dir, "%d.tar" % (22))
        if os.path.isfile(outfile):
            saved_states = torch.load(outfile)
            state_dict = self.clfs.state_dict()
            state_dict.update(saved_states)
            self.clfs.load_state_dict(state_dict)
        else:
            self.train_d_specific_classifiers(n_clf)

    def train_d_specific_classifiers(self, n_clf):
        in_dim = int(self.feat_dim / n_clf)
        out_dim = self.num_classes
        self.clfs = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(n_clf)])
        self.clfs = self.clfs.cuda()
        tiered_mini = False
        if self.dataset == "tiered":
            tiered_mini = True
            base_file = "base"
        else:
            base_file = configs.data_dir['miniImagenet'] + "base" + '.json'
        batch_size = 128
        base_datamgr = SimpleDataManager(self.image_size, batch_size=batch_size)
        base_loader = base_datamgr.get_data_loader(base_file, aug=True, num_workers=12, tiered_mini=tiered_mini)
        loss_fn = nn.CrossEntropyLoss()
        params = self.clfs.parameters()
        optimizer = torch.optim.Adam(params)
        for epoch in range(0, 40):
            for i, (x, y, _) in enumerate(base_loader):
                optimizer.zero_grad()
                x = Variable(x.cuda())
                out = self.get_features(x)
                y = y.cuda()
                avg_loss = 0
                for j in range(n_clf):
                    start = in_dim * j
                    stop = start + in_dim
                    scores = self.clfs[j](out[:, start:stop])
                    loss = loss_fn(scores, y)
                    loss.backward(retain_graph=True)
                    avg_loss += loss.item()
                optimizer.step()
                if i % 10 == 0:
                    print("Epoch: %d, Batch %d/%d, Loss=%.3f" % (epoch, i, len(base_loader), avg_loss / n_clf))
            # save model
            out_dir = "pretrain/clfs/%s_%s_%s_%d" % (self.method, self.model_name, self.base_dataset, n_clf)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            outfile = os.path.join(out_dir, "%d.tar" % (epoch))
            torch.save(self.clfs.state_dict(), outfile)

    def test_d_specific_classifiers(self, n_clf):
        in_dim = int(self.feat_dim / n_clf)
        out_dim = self.num_classes
        tiered_mini = False
        if self.dataset == "tiered":
            tiered_mini = True
            base_file = "base"
        else:
            base_file = configs.data_dir['miniImagenet'] + "base" + '.json'
        batch_size = 32
        base_datamgr = SimpleDataManager(self.image_size, batch_size=batch_size)
        base_loader = base_datamgr.get_data_loader(base_file, aug=False, num_workers=0, tiered_mini=tiered_mini)
        correct_counts = np.zeros(n_clf)
        total = 0
        for i, (x, y, _) in enumerate(base_loader):
            x = Variable(x.cuda())
            out = self.get_features(x)
            total += out.shape[0]
            for j in range(n_clf):
                start = in_dim * j
                stop = start + in_dim
                scores = self.clfs[j](out[:, start:stop])
                pred = scores.data.cpu().numpy().argmax(axis=1)
                y_np = y.cpu().numpy()
                correct_counts[j] += (pred == y_np).sum()
        correct_counts = correct_counts / total

    def save_pretrain_dataset(self, split):
        params = self.params
        tiered_mini = False
        if params.dataset == 'cross':
            # base_file = configs.data_dir['miniImagenet'] + 'all.json'  # Original code
            if split == "base":
                base_file = configs.data_dir['miniImagenet'] + split + '.json'
            elif split == "novel":
                base_file = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        elif params.dataset == "tiered":
            base_file = split
            tiered_mini = True
        else:
            base_file = configs.data_dir[params.dataset] + split + '.json'

        batch_size = self.batch_size

        base_datamgr = SimpleDataManager(self.image_size, batch_size=batch_size)
        base_loader = base_datamgr.get_data_loader(base_file, aug=False, num_workers=12, tiered_mini=tiered_mini)

        features = []
        labels = []
        print("Saving pretrain dataset...")
        for epoch in range(0, 1):
            for i, (x, y, _) in enumerate(base_loader):
                x = Variable(x.cuda())
                out = self.get_features(x)
                current_batch_size = out.shape[0]
                for j in range(current_batch_size):
                    np_out = out.data.cpu().numpy()[j]
                    np_y = y.numpy()[j]
                    features.append(np_out)
                    labels.append(np_y)
                print_with_carriage_return("Epoch %d: %d/%d processed" % (epoch, i, len(base_loader.dataset.meta["image_labels"]) / batch_size))
            end_carriage_return_print()
        dataset = self.params.dataset
        features_dir = "pretrain/features_%s_%s_%s_%s.npy" % (dataset, self.params.method, self.model_name, split)
        labels_dir = "pretrain/labels_%s_%s_%s_%s.npy" % (dataset, self.params.method, self.model_name, split)
        np.save(features_dir, np.asarray(features))
        np.save(labels_dir, np.asarray(labels))
        return np.asarray(features), np.asarray(labels)

    def get_pretrain_dataset(self, split):
        dataset = self.params.dataset
        features_dir = "pretrain/features_%s_%s_%s_%s.npy" % (dataset, self.params.method, self.model_name, split)
        labels_dir = "pretrain/labels_%s_%s_%s_%s.npy" % (dataset, self.params.method, self.model_name, split)
        if os.path.isfile(features_dir) and os.path.isfile(labels_dir):
            features = np.load(features_dir)
            labels = np.load(labels_dir)
        else:
            features, labels = self.save_pretrain_dataset(split)
        return features, labels

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def _calc_pretrained_class_mean(self, normalize=False):
        params = self.params
        print(params)
        aug = False
        tiered_mini = False
        if params.dataset == 'cross':
            # base_file = configs.data_dir['miniImagenet'] + 'all.json'  # Original code
            base_file = configs.data_dir['miniImagenet'] + 'base.json'
        elif params.dataset == 'cross_char':
            base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        elif params.dataset == "tiered":
            base_file = "base"
            tiered_mini = True
        else:
            base_file = configs.data_dir[params.dataset] + 'base.json'

        batch_size = self.batch_size
        base_datamgr = SimpleDataManager(self.image_size, batch_size=batch_size)
        # No aug for every method
        base_loader = base_datamgr.get_data_loader(base_file, aug=aug, num_workers=12, tiered_mini=tiered_mini)

        features = np.zeros((self.num_classes, self.feat_dim))
        counts = np.zeros(self.num_classes)
        print("saving pretrained mean")
        for epoch in range(0, 1):
            for i, (x, y, path) in enumerate(base_loader):
                x = Variable(x.cuda())
                out = self.get_features(x)
                if normalize:
                    out = self.normalize(out)
                for j in range(out.shape[0]):
                    np_out = out.data.cpu().numpy()[j]
                    np_y = y.numpy()[j]
                    features[np_y] += np_out
                    counts[np_y] += 1
                print_with_carriage_return("Epoch %d: %d/%d processed" % (epoch, i, len(base_loader)))
            end_carriage_return_print()
            # print(np.max(counts[64:]))
            print(np.max(features))
        for i in range(0, len(counts)):
            if counts[i] != 0:
                features[i] = features[i] / counts[i]
        return features

    def get_kmeans_pca_model(self, k=8, n_clusters=10, normalize=False):
        if normalize:
            pre = "norm_"
        else:
            pre = ""
        kmeans_saved_model_dir = "pretrain/%s%s_%s_%s_%s_k%s.sav" % (pre, self.params.method, self.dataset, self.params.model, str(n_clusters), str(k))
        pca_saved_model_dir = "pretrain/pca_%s%s_%s_%s_%s.sav" % (pre, self.params.method, self.dataset, self.params.model, str(k))
        
        features, _ = self.get_pretrain_dataset("base")
        if normalize:
            features_pt = torch.from_numpy(features).cuda()
            features_pt = self.normalize(features_pt)
            features = features_pt.cpu().numpy()
            
        if os.path.isfile(kmeans_saved_model_dir):
            kmeans_model = pickle.load(open(kmeans_saved_model_dir, 'rb'))
            if k > 0:
                pca_model = pickle.load(open(pca_saved_model_dir, 'rb'))
            else:
                pca_model = None
        else:
            # PCA for dimension reduction
            '''
            # Result is different from sklearn PCA. This indicates my PCA code is problematic??
            features_cuda = torch.from_numpy(features).cuda()
            U, S, V, X_mean = pca.SVD(features_cuda)
            weight = V[:, :k]
            transformed = torch.mm(features_cuda, weight)
            transformed_np = transformed.cpu().numpy()
            '''
            # sklearn pca for verification
            if k <= 0:
                sk_transformed = features
                pca_model = None
            else:
                pca_model = PCA(n_components=k)
                sk_transformed = pca_model.fit_transform(features)

            print("Training KMeans model for %s with %s clusters and k=%s" % (self.params.method, str(n_clusters), str(k)))
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(sk_transformed)
            # save model
            pickle.dump(kmeans_model, open(kmeans_saved_model_dir, 'wb'))
            if k > 0:
                pickle.dump(pca_model, open(pca_saved_model_dir, 'wb'))
        # Calculate base cluster means
        labels = kmeans_model.labels_
        self.new_cluster_means = torch.zeros(n_clusters, self.feat_dim).cuda()
        features = torch.from_numpy(features).cuda()
        for i in range(n_clusters):
            self.new_cluster_means[i] = features[labels == i].mean(dim=0)
        return pca_model, kmeans_model
    
    def get_pretrained_class_mean(self, normalize=False):
        if normalize:
            pre = "norm_"
        else:
            pre = ""
        save_dir = "pretrain/%s%s_%s_%s_mean.npy" % (pre, self.base_dataset, self.params.method, self.params.model)
        if os.path.isfile(save_dir):
            features = np.load(save_dir)
        else:
            # normalize = False
            features = self._calc_pretrained_class_mean(normalize=normalize)
            np.save(save_dir, features)
        return features
