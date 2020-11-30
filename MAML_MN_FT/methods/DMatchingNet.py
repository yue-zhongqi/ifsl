# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
import utils
from methods.meta_toolkits import MatchingNetModule
from methods.meta_toolkits import FeatureProcessor


class DMatchingNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, pretrain, n_splits, is_cosine_feature, d_feature, fusion, sum_log,
                 classifier="single", use_counterfactual=False, x_zero=False, num_classes=64, logit_fusion="product",
                 temp=100, use_x_only=False, preprocess_after_split="none",
                 preprocess_before_split="none", normalize_before_center=False, normalize_d=False, normalize_ed=False):
        super(DMatchingNet, self).__init__(model_func, n_way, n_support, image_size=84)
        self.n_splits = n_splits
        self.is_cosine_feature = is_cosine_feature
        self.d_feature = d_feature
        self.pretrain = pretrain
        self.num_classes = num_classes
        self.fusion = fusion
        self.sum_log = sum_log
        self.classifier = classifier
        self.use_counterfactual = use_counterfactual
        self.x_zero = x_zero
        self.logit_fusion = logit_fusion
        self.temp = temp
        self.use_x_only = use_x_only
        self.preprocess_after_split = preprocess_after_split
        self.preprocess_before_split = preprocess_before_split
        self.normalize_before_center = normalize_before_center
        self.normalize_d = normalize_d
        self.normalize_ed = normalize_ed

        if self.use_x_only:
            self.classifier = "bi"

        self.loss_fn = nn.NLLLoss()
        if self.classifier == "single":
            feat_dim = self.get_feat_dim()
            self.matching_nets = nn.ModuleList([MatchingNetModule(feat_dim).cuda() for i in range(n_splits)])
        elif self.classifier == "bi":
            x_feat_dim = int(self.feat_dim / self.n_splits)
            if self.d_feature == "pd":
                d_feat_dim = self.num_classes
            else:
                d_feat_dim = x_feat_dim
            self.x_matching_nets = nn.ModuleList([MatchingNetModule(x_feat_dim).cuda() for i in range(n_splits)])
            self.d_matching_nets = nn.ModuleList([MatchingNetModule(d_feat_dim).cuda() for i in range(n_splits)])
        self.feature_processor = FeatureProcessor(self.pretrain, self.n_splits, self.is_cosine_feature, self.d_feature, self.num_classes,
                                                  preprocess_after_split=preprocess_after_split, preprocess_before_split=preprocess_before_split,
                                                  normalize_before_center=normalize_before_center, normalize_d=normalize_d, normalize_ed=normalize_ed)
        self.relu = nn.ReLU()
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

    def set_forward(self, x, is_feature=False):
        support, query = self.parse_feature(x, is_feature)
        support = support.contiguous().view(self.n_way * self.n_support, -1 )
        query = query.contiguous().view(self.n_way * self.n_query, -1 )
        
        split_support, support_d, split_query, query_d = self.feature_processor.get_features(support, query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)

        scores = torch.zeros(self.n_splits, query.shape[0], support.shape[0]).cuda()
        c_scores = torch.zeros(self.n_splits, query.shape[0], support.shape[0]).cuda()
        if self.x_zero:
            c_split_query = torch.zeros_like(split_query).cuda()
        else:
            c_split_query = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
        c_fused_query = self.fuse_features(c_split_query, query_d)
        for i in range(self.n_splits):
            if self.classifier == "single":
                support_new, query_new = self.matching_nets[i](fused_support[i], fused_query[i])
                _, c_query_new = self.matching_nets[i](fused_support[i], c_fused_query[i])
                scores[i] = self.relu(self.normalize(query_new).mm(self.normalize(support_new).transpose(0, 1))) * self.temp
                c_scores[i] = self.relu(self.normalize(c_query_new).mm(self.normalize(support_new).transpose(0, 1))) * self.temp
            elif self.classifier == "bi":
                support_x_new, query_x_new = self.x_matching_nets[i](split_support[i], split_query[i])
                support_d_new, query_d_new = self.d_matching_nets[i](support_d[i], query_d[i])
                x_score = self.relu(self.normalize(query_x_new).mm(self.normalize(support_x_new).transpose(0, 1)))
                d_score = self.relu(self.normalize(query_d_new).mm(self.normalize(support_d_new).transpose(0, 1)))
                c_x_scores = torch.ones_like(x_score).cuda()
                if self.use_x_only:
                    scores[i] = x_score * self.temp
                    c_scores[i] = c_x_scores * self.temp
                else:
                    scores[i] = self.fuse_proba(x_score, d_score) * self.temp
                    c_scores[i] = self.fuse_proba(c_x_scores, d_score) * self.temp
        if self.use_counterfactual:
            scores = scores - c_scores
        scores = self.softmax(scores)
        labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        labels = Variable(utils.one_hot(labels, self.n_way)).cuda()

        if self.sum_log:
            # not working well
            logscores = (scores.matmul(labels) + 1e-6)
            logscores = scores.log()
            logprobs = logscores.mean(dim=0)
        else:
            proba = scores.mean(dim=0)
            logprobs = (proba.mm(labels) + 1e-6).log()
        return logprobs

    def predict(self, support, query):
        split_support, support_d, split_query, query_d = self.feature_processor.get_features(support, query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)
        scores = torch.zeros(self.n_splits, query.shape[0], support.shape[0]).cuda()
        for i in range(self.n_splits):
            support_new, query_new = self.matching_nets[i](fused_support[i], fused_query[i])
            scores[i] = self.relu(self.normalize(query_new).mm(self.normalize(support_new).transpose(0, 1))) * self.temp
        scores = self.softmax(scores)
        labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        labels = Variable(utils.one_hot(labels, self.n_way)).cuda()
        proba = scores.mean(dim=0)
        logprobs = (proba.mm(labels) + 1e-6).log()
        return logprobs

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way ), self.n_query))
        y_query = Variable(y_query.cuda())
        logprobs = self.set_forward(x)
        self.current_scores = logprobs
        return self.loss_fn(logprobs, y_query)

