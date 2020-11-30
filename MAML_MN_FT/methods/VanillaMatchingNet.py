# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
import utils
from methods.meta_toolkits import MatchingNetModule


class VanillaMatchingNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(VanillaMatchingNet, self).__init__(model_func, n_way, n_support, image_size=84)
        self.loss_fn = nn.NLLLoss()
        self.matching_net_module = MatchingNetModule(self.feat_dim).cuda()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def set_forward(self, x, is_feature=False):
        support, query = self.parse_feature(x, is_feature)
        support = support.contiguous().view(self.n_way * self.n_support, -1)
        query = query.contiguous().view(self.n_way * self.n_query, -1)
        
        support, query = self.matching_net_module(support, query)

        labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        labels = Variable(utils.one_hot(labels, self.n_way)).cuda()

        scores = self.relu(self.normalize(query).mm(self.normalize(support).transpose(0, 1))) * 100
        proba = self.softmax(scores)
        logprobs = (proba.mm(labels) + 1e-6).log()
        return logprobs

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way ), self.n_query))
        y_query = Variable(y_query.cuda())
        logprobs = self.set_forward(x)
        self.current_scores = logprobs
        return self.loss_fn(logprobs, y_query)

