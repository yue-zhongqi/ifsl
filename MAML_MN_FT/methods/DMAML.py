# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.meta_toolkits import MAMLBlock
from methods.meta_toolkits import FeatureProcessor


class DMAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, pretrain, n_splits, is_cosine_feature, d_feature, fusion, sum_log,
                 approx, update_step, n_task, lr, classifier="single", use_counterfactual=False, x_zero=False,
                 num_classes=64, temp=5, logit_fusion="product", use_x_only=False, preprocess_after_split="none",
                 preprocess_before_split="none", normalize_before_center=False, normalize_d=False, normalize_ed=False):
        super(DMAML, self).__init__(model_func, n_way, n_support, change_way=False)
        self.n_splits = n_splits
        self.is_cosine_feature = is_cosine_feature
        self.d_feature = d_feature
        self.pretrain = pretrain
        self.num_classes = num_classes
        self.fusion = fusion
        self.sum_log = sum_log
        self.n_task = n_task
        self.update_step = update_step
        self.lr = lr
        self.approx = approx
        self.n_task = n_task
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

        self.feature_processor = FeatureProcessor(self.pretrain, self.n_splits, self.is_cosine_feature, self.d_feature, self.num_classes,
                                                  preprocess_after_split=preprocess_after_split, preprocess_before_split=preprocess_before_split,
                                                  normalize_before_center=normalize_before_center, normalize_d=normalize_d, normalize_ed=normalize_ed)
                                                  
        if self.classifier == "single":
            feat_dim = self.get_feat_dim()
            self.maml_blocks = nn.ModuleList([MAMLBlock(feat_dim, self.n_way, update_step, approx, lr) for i in range(n_splits)])
        elif self.classifier == "bi":
            x_feat_dim = int(self.feat_dim / self.n_splits)
            if self.d_feature == "pd":
                d_feat_dim = self.num_classes
            else:
                d_feat_dim = x_feat_dim
            self.x_maml_blocks = nn.ModuleList([MAMLBlock(x_feat_dim, self.n_way, update_step, approx, lr) for i in range(n_splits)])
            self.d_maml_blocks = nn.ModuleList([MAMLBlock(d_feat_dim, self.n_way, update_step, approx, lr) for i in range(n_splits)])
        self.loss_fn = nn.NLLLoss()
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
        support = support.contiguous().view(self.n_way * self.n_support, -1)
        query = query.contiguous().view(self.n_way * self.n_query, -1)
        labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

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
                self.maml_blocks[i].fit(fused_support[i], labels)
                scores[i] = self.maml_blocks[i].predict(fused_query[i]) * self.temp
                c_scores[i] = self.maml_blocks[i].predict(c_fused_query[i]) * self.temp
            elif self.classifier == "bi":
                self.x_maml_blocks[i].fit(split_support[i], labels)
                self.d_maml_blocks[i].fit(support_d[i], labels)
                x_score = self.x_maml_blocks[i].predict(split_query[i])
                d_score = self.d_maml_blocks[i].predict(query_d[i])
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

    def set_forward_adaptation(self, x, is_feature=False):
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        self.current_scores = scores
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way ), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            loss = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)
            task_count += 1

            if task_count == self.n_task:
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
                      
    def test_loop(self, test_loader, return_std=False, metric="acc"):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x, metric=metric)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

