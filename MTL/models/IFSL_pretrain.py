import os
from models.ResNet10 import ResNet10
from models.WRN28 import WideRes28
import torch
import numpy as np
import tqdm
from dataloader.dataset_loader import DatasetLoader as Dataset
from torch.utils.data import DataLoader


class Pretrain():
    def __init__(self, dataset, method, model, init_model=True):
        self.dataset = dataset
        self.method = method
        self.model_name = model
        self.init_model = init_model
        if self.method in ["simpleshot", "simpleshotwide"]:
            self.simpleshot_init()
        for param in self.model.parameters():
            param.requires_grad = False

    def simpleshot_init(self):
        simple_shot_dir = "/data2/yuezhongqi/Model/simple_shot/"
        model_name = self.model_name.lower()
        if self.dataset == "tiered":
            self.num_classes = 351
        else:
            self.num_classes = 64
        self.image_size = 84
        if model_name == "wideres":
            self.batch_size = 128
            self.feat_dim = 640
        else:
            self.batch_size = 128
            self.feat_dim = 512

        if model_name == "resnet10":
            model_abbr = "resnet"
        elif model_name == "wideres":
            model_abbr = "wrn"
        if self.dataset == "cross" or self.dataset == "miniImagenet":
            model_dir = os.path.join(simple_shot_dir, "miniImagenet", model_name, "model_best.pth.tar")
            # model_dir = "/model/1154027137/ifsl_mini_pretrain/ifsl_mini/ss_" + model_abbr + "_mini.tar"
        elif self.dataset == "tiered":
            model_dir = os.path.join(simple_shot_dir, "tiered", model_name, "model_best.pth.tar")
            # model_dir = "/model/1154027137/ifsl_tiered_pretrain/ifsl_tiered/ss_" + model_abbr + "_tiered.tar"

        def remove_module_from_param_name(params_name_str):
            split_str = params_name_str.split(".")[1:]
            params_name_str = ".".join(split_str)
            return params_name_str

        if self.init_model:
            if model_name == "resnet10":
                model = ResNet10(num_classes=self.num_classes, remove_linear=False)
            elif model_name == "wideres":
                model = WideRes28(num_classes=self.num_classes, remove_linear=False)
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
        return prob

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

    def get_pretrained_class_mean(self, normalize=False):
        if normalize:
            pre = "norm_"
        else:
            pre = ""
        save_dir = "pretrain/%s%s_%s_%s_mean.npy" % (pre, self.dataset, self.method, self.model_name)
        if os.path.isfile(save_dir):
            features = np.load(save_dir)
        else:
            # normalize = False
            features = self.get_base_means(normalize=normalize)
            np.save(save_dir, features)
        return features

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def get_base_means(self, normalize=False):
        num_classes = self.num_classes
        # save_dir = "pretrain/%s%s_%s_%s_mean.npy" % (pre, self.dataset, self.method, self.model_name)
        # ensure_path("pretrain")
        # self.means_save_dir = osp.join("logs/means", "%s_%s.npy" % (self.args.dataset, str(is_cosine_feature)))
        # Load pretrain set
        num_workers = 8
        if self.args.debug:
            num_workers = 0
        self.trainset = Dataset('train', self.args, dataset=self.dataset, train_aug=False)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.args.pre_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        means = torch.zeros(num_classes, 640).cuda()
        counts = torch.zeros(num_classes).cuda()
        for epoch in range(1):
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, _ = [_.cuda() for _ in batch]
                label = batch[1]
                with torch.no_grad():
                    data = self.encoder(data)
                if normalize:
                    data = self.normalize(data)
                for j in range(data.shape[0]):
                    means[label[j]] += data[j]
                    counts[label[j]] += 1
        counts = counts.unsqueeze(1).expand_as(means)
        means = means / counts
        means_np = means.cpu().detach().numpy()
        # np.save(save_dir, means_np)
        return means_np