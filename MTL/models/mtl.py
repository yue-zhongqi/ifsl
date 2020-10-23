##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl
from models.ResNet10 import ResNet10MTL
from models.WRN28 import WideRes28Mtl
import numpy as np
from models.IFSL import PretrainNet
from models.IFSL import DeconfoundedLearner
from models.IFSL_modules import IFSLBaseLearner
from models.IFSL_pretrain import Pretrain


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

    def initialize(self):
        torch.nn.init.kaiming_normal_(self.fc1_w)
        torch.nn.init.constant_(self.fc1_b, 0)


class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.pretrain = Pretrain(args.param.dataset, args.param.method, args.param.model, True)
        z_dim = self.pretrain.feat_dim
        num_classes = self.pretrain.num_classes
        if not args.deconfound:
            self.base_learner = BaseLearner(args, z_dim)
        else:
            if args.param.learner == "IFSL":
                self.base_learner = IFSL(args.way, args.shot, self.pretrain, **args.param.ifsl_params)
            else:
                self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            # self.encoder = ResNetMtl()
            # hard code for ResNet10 as of now
            num_classes = num_classes
            if args.param.model == "ResNet10":
                self.encoder = ResNet10MTL(num_classes=num_classes, remove_linear=False)
            elif args.param.model == "wideres":
                self.encoder = WideRes28Mtl(num_classes=num_classes, remove_linear=False)
        else:
            self.encoder = ResNetMtl(mtl=False)
            self.pre_fc = nn.Sequential(nn.Linear(640, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
        # self.pretrain = PretrainNet(args)

    def load_pretrain_weight(self, model_dir):
        # hard code for ResNet10 as of now
        model_params = torch.load(self.args.init_weights)['state_dict']

        def remove_module_from_param_name(params_name_str):
            split_str = params_name_str.split(".")[1:]
            params_name_str = ".".join(split_str)
            return params_name_str

        model_params = {remove_module_from_param_name(k): v for k, v in model_params.items()}
        model_dict = self.encoder.state_dict()
        matched_dict = {k: v for k, v in model_params.items() if k in model_dict}
        # print(matched_dict.keys())
        model_dict.update(model_params)
        self.encoder.load_state_dict(model_dict)

    def encode(self, x):
        # hard code for ResNet10 as of now
        return self.encoder(x, feature=True)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query, val = inp
            return self.meta_forward(data_shot, label_shot, data_query, val)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query, val=False):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """

        cat_features = torch.cat((data_shot, data_query), dim=0)
        n_shot = data_shot.shape[0]
        if val:
            with torch.no_grad():
                embeddings, _ = self.encode(cat_features)
        else:
            embeddings, _ = self.encode(cat_features)
        embedding_shot = embeddings[:n_shot]
        embedding_query = embeddings[n_shot:]
        if not self.args.deconfound:
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, self.base_learner.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
            for _ in range(1, self.update_step):
                logits = self.base_learner(embedding_shot, fast_weights)
                loss = F.cross_entropy(logits, label_shot)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            self.fast_weights = fast_weights
            self.embedding_shot = embedding_shot
            self.embedding_query = embedding_query
            logits_q = self.base_learner(embedding_query, fast_weights)
        else:
            learner = DeconfoundedLearner(pretrain=self.pretrain, **self.args.param.ifsl_params)
            logits_q = learner.fit(data_shot, data_query, label_shot, embedding_shot, embedding_query)
            self.learner = learner
            self.embedding_shot = embedding_shot
            self.embedding_query = embedding_query
        return logits_q

    def predict(self, embedding_shot, label_shot, embedding_query):
        if not self.args.deconfound:
            return self.base_learner(embedding_query, self.fast_weights)
        else:
            return self.learner.predict(label_shot, embedding_shot, embedding_query)

    def backward_loss_and_step(self, loss, optimizer=None):
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        for i, param in enumerate(self.base_learner.parameters()):
            param.grad = grad[i]
            # param.data.add_(- self.update_lr * grad[i])
        optimizer.step()
        
    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q