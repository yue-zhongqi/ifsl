# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.meta_toolkits import MAMLBlock


class VanillaMAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False, update_step=10, n_task=4, lr=0.01):
        super(VanillaMAML, self).__init__(model_func, n_way, n_support, change_way=False)
        self.n_task = n_task
        self.update_step = update_step
        self.lr = lr
        self.approx = approx
        self.n_task = n_task

        self.maml_block = MAMLBlock(self.feat_dim, self.n_way, update_step, approx, lr).cuda()
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        support, query = self.parse_feature(x, is_feature)
        support = support.contiguous().view(self.n_way * self.n_support, -1)
        query = query.contiguous().view(self.n_way * self.n_query, -1)
        labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()
        self.maml_block.fit(support, labels)
        return self.maml_block.predict(query)

    def set_forward_adaptation(self, x, is_feature=False):
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
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

