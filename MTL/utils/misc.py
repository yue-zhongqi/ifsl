##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Additional utility functions. """
import os
import time
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized

def count_dacc(pred_logits, support_labels, query_labels, support_imgs, query_imgs, pretrain):
    relu = torch.nn.ReLU()
    softmax = torch.nn.Softmax(dim=1)
    num_classes = pretrain.num_classes
    n_way = 5
    with torch.no_grad():
        support = pretrain.get_features(support_imgs)
        query = pretrain.get_features(query_imgs)
        support_probs = relu(pretrain.classify(support, normalize_prob=False)[:, :num_classes])
        query_probs = relu(pretrain.classify(query, normalize_prob=False)[:, :num_classes])
    w = torch.zeros(n_way, support_probs.shape[1]).cuda()
    for i in range(n_way):
        w[i] = support_probs[support_labels == i].mean(dim=0)
    w = normalize(w)
    query_logits = normalize(query_probs)
    w = w.unsqueeze(0).expand(query_logits.shape[0], -1, -1)
    query_logits = query_logits.unsqueeze(1).expand(-1, n_way, -1)
    logits = (w * query_logits).sum(dim=2)  # 75 * 5
    query_probs = softmax(logits)
    hardness = []
    for i in range(query_probs.shape[0]):
        p = query_probs[i][query_labels[i]]
        log_odd = torch.log((1 - p) / p)
        hardness.append(log_odd.cpu().numpy())
    hardness = np.array(hardness)
    if hardness.min() < 0:
        hardness -= hardness.min()
    pred = F.softmax(pred_logits, dim=1).argmax(dim=1)
    return ((pred == query_labels).detach().cpu().numpy() * hardness).sum() / hardness.sum()

def get_hardness_correct(pred_logits, support_labels, query_labels, support_imgs, query_imgs, pretrain):
    relu = torch.nn.ReLU()
    softmax = torch.nn.Softmax(dim=1)
    num_classes = pretrain.num_classes
    n_way = 5
    with torch.no_grad():
        support = pretrain.get_features(support_imgs)
        query = pretrain.get_features(query_imgs)
        support_probs = relu(pretrain.classify(support, normalize_prob=False)[:, :num_classes])
        query_probs = relu(pretrain.classify(query, normalize_prob=False)[:, :num_classes])
    w = torch.zeros(n_way, support_probs.shape[1]).cuda()
    for i in range(n_way):
        w[i] = support_probs[support_labels == i].mean(dim=0)
    w = normalize(w)
    query_logits = normalize(query_probs)
    w = w.unsqueeze(0).expand(query_logits.shape[0], -1, -1)
    query_logits = query_logits.unsqueeze(1).expand(-1, n_way, -1)
    logits = (w * query_logits).sum(dim=2)  # 75 * 5
    query_probs = softmax(logits)
    hardness = []
    for i in range(query_probs.shape[0]):
        p = query_probs[i][query_labels[i]]
        log_odd = torch.log((1 - p) / p)
        hardness.append(log_odd.cpu().numpy())
    hardness = np.array(hardness)
    pred = F.softmax(pred_logits, dim=1).argmax(dim=1)
    return hardness, (pred == query_labels).detach().cpu().numpy()

class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()