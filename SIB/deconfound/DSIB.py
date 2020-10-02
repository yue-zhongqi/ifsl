# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sib import ClassifierSIB
from deconfound.meta_toolkits import FeatureProcessor


class DeconfoundedSIB(nn.Module):
    def __init__(self, n_way, pretrain, n_splits, is_cosine_feature, d_feature, fusion, sum_log,
                 lr, classifier="single", use_counterfactual=False, x_zero=False, q_steps=3,
                 num_classes=64, temp=1, logit_fusion="product", use_x_only=False, feat_dim=640,
                 preprocess_after_split="none", preprocess_before_split="none", normalize_before_center=False,
                 normalize_d=False, normalize_ed=False):
        super(DeconfoundedSIB, self).__init__()
        self.n_way = n_way
        self.n_splits = n_splits
        self.is_cosine_feature = is_cosine_feature
        self.d_feature = d_feature
        self.pretrain = pretrain
        self.num_classes = num_classes
        self.fusion = fusion
        self.sum_log = sum_log
        self.lr = lr
        self.classifier = classifier
        self.use_counterfactual = use_counterfactual
        self.x_zero = x_zero
        self.logit_fusion = logit_fusion
        self.temp = temp
        self.use_x_only = use_x_only
        self.q_steps = 3
        self.feat_dim = feat_dim
        self.preprocess_after_split = preprocess_after_split
        self.preprocess_before_split = preprocess_before_split
        self.normalize_before_center = normalize_before_center
        self.normalize_d = normalize_d
        self.normalize_ed = normalize_ed

        if self.use_x_only:
            self.classifier = "bi"

        self.feature_processor = FeatureProcessor(self.pretrain, self.n_splits, self.is_cosine_feature, self.d_feature, self.num_classes,
                                                  self.preprocess_after_split, self.preprocess_before_split, self.normalize_before_center,
                                                  self.normalize_d, self.normalize_ed)
        if self.classifier == "single":
            feat_dim = self.get_feat_dim()
            self.sib_blocks = nn.ModuleList([ClassifierSIB(self.n_way, feat_dim, self.q_steps) for i in range(n_splits)])
        elif self.classifier == "bi":
            x_feat_dim = int(self.feat_dim / self.n_splits)
            if self.d_feature == "pd":
                d_feat_dim = self.num_classes
            else:
                d_feat_dim = x_feat_dim
            self.x_sib_blocks = nn.ModuleList([ClassifierSIB(self.n_way, x_feat_dim, self.q_steps) for i in range(n_splits)])
            self.d_sib_blocks = nn.ModuleList([ClassifierSIB(self.n_way, d_feat_dim, self.q_steps) for i in range(n_splits)])
        self.softmax = nn.Softmax(dim=2)

    def get_feat_dim(self):
        split_feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "pd":
            return split_feat_dim + self.num_classes
        else:
            if self.fusion == "concat":
                return split_feat_dim * 2
            else:
                return split_feat_dim

    def fuse_features(self, x1, x2):
        if self.fusion == "concat":
            return torch.cat((x1, x2), dim=2)
        elif self.fusion == "+":
            return x1 + x2
        elif self.fusion == "-":
            return x1 - x2

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def fuse_proba(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == "linear_sum":
            return p1 + p2
        elif self.logit_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def forward(self, support, labels, query, _):
        support = support.squeeze(0)
        query = query.squeeze(0)
        split_support, support_d, split_query, query_d = self.feature_processor.get_features(support, query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)

        scores = torch.zeros(self.n_splits, query.shape[0], self.n_way).cuda()
        c_scores = torch.zeros(self.n_splits, query.shape[0], self.n_way).cuda()

        if self.x_zero:
            c_split_query = torch.zeros_like(split_query).cuda()
        else:
            c_split_query = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
        c_fused_query = self.fuse_features(c_split_query, query_d)

        for i in range(self.n_splits):
            if self.classifier == "single":
                scores[i] = self.sib_blocks[i](fused_support[i].unsqueeze(0), labels, fused_query[i].unsqueeze(0), self.lr) * self.temp
                c_scores[i] = self.sib_blocks[i](fused_support[i].unsqueeze(0), labels, c_fused_query[i].unsqueeze(0), self.lr) * self.temp
            elif self.classifier == "bi":
                x_score = self.x_sib_blocks[i](split_support[i].unsqueeze(0), labels, split_query[i].unsqueeze(0), self.lr)
                d_score = self.d_sib_blocks[i](support_d[i].unsqueeze(0), labels, query_d[i].unsqueeze(0), self.lr)
                c_x_scores = torch.ones_like(x_score).cuda()
                if self.use_x_only:
                    scores[i] = x_score * self.temp
                    c_scores[i] = c_x_scores * self.temp
                else:
                    scores[i] = self.fuse_proba(x_score, d_score) * self.temp
                    c_scores[i] = self.fuse_proba(c_x_scores, d_score) * self.temp
        scores = self.softmax(scores)
        c_scores = self.softmax(c_scores)
        if self.use_counterfactual:
            scores = scores - c_scores
        if self.sum_log:
            scores = scores.log()
            scores = scores.mean(dim=0)
        else:
            scores = scores.mean(dim=0)
            scores = scores.log()
        return scores