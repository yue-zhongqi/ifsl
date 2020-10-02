# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import os
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device
from PretrainedModel import PretrainedModel
import numpy as np


class Params:
    def __init__(self):
        self.dummy = True

class Algorithm:
    """
    Algorithm logic is implemented here with training and validation functions etc.

    :param args: experimental configurations
    :type args: EasyDict
    :param logger: logger
    :param netFeat: feature network
    :type netFeat: class `WideResNet` or `ConvNet_4_64`
    :param netSIB: Classifier/decoder
    :type netSIB: class `ClassifierSIB`
    :param optimizer: optimizer
    :type optimizer: torch.optim.SGD
    :param criterion: loss
    :type criterion: nn.CrossEntropyLoss
    """
    def __init__(self, args, logger, netFeat, netSIB, optimizer, criterion, pretrain=None):
        self.netFeat = netFeat
        self.netSIB = netSIB
        self.optimizer = optimizer
        self.criterion = criterion

        self.nbIter = args.nbIter
        self.nStep = args.nStep
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.batchSize = args.batchSize
        self.nEpisode = args.nEpisode
        self.momentum = args.momentum
        self.weightDecay = args.weightDecay

        self.davg = args.davg

        self.logger = logger
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Loading pretrained netFeat is done in main.py
        '''
        if args.resumeFeatPth :
            if args.cuda:
                param = torch.load(args.resumeFeatPth)
            else:
                param = torch.load(args.resumeFeatPth, map_location='cpu')
            self.netFeat.load_state_dict(param)
            msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
            self.logger.info(msg)
        '''

        if args.test:
            self.load_ckpt(args.ckptPth)

        # dfsl pretrain
        self.pretrain = pretrain

    def load_ckpt(self, ckptPth):
        """
        Load checkpoint from ckptPth.

        :param ckptPth: the path to the ckpt
        :type ckptPth: string
        """
        param = torch.load(ckptPth)
        self.netFeat.load_state_dict(param['netFeat'])
        self.netSIB.load_state_dict(param['SIB'])
        lr = param['lr']
        self.optimizer = torch.optim.SGD(itertools.chain(*[self.netSIB.parameters(),]),
                                         lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weightDecay,
                                         nesterov=True)
        msg = '\nLoading networks from {}'.format(ckptPth)
        self.logger.info(msg)


    def compute_grad_loss(self, clsScore, QueryLabel):
        """
        Compute the loss between true gradients and synthetic gradients.
        """
        # register hooks
        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g
            h = v.register_hook(hook)
            return h
        handle = require_nonleaf_grad(clsScore)

        loss = self.criterion(clsScore, QueryLabel)
        loss.backward(retain_graph=True) # need to backward again

        # remove hook
        handle.remove()

        gradLogit = self.netSIB.dni(clsScore) # B * n x nKnovel
        gradLoss = F.mse_loss(gradLogit, clsScore.grad_nonleaf.detach())

        return loss, gradLoss

    def cosine_similarity(self, a, b):
        return (a * b).sum() / torch.norm(a, p=2) / torch.norm(b, p=2)

    def calc_diff_scores(self, pretrain, support, query, support_labels, query_labels):
        support_probs = pretrain.classify(support)
        query_probs = pretrain.classify(query)
        total_diff_scores = []
        for i in range(query.shape[0]):
            current_query_diff_score = 0
            label = query_labels[i]
            for j in range(support.shape[0]):
                if support_labels[j] == label:
                    similarity = self.cosine_similarity(query_probs[i], support_probs[j])
                    current_query_diff_score += (1 - similarity)
            current_query_diff_score /= int(support.shape[0] / 5)
            total_diff_scores.append(current_query_diff_score.cpu().numpy())
        return np.array(total_diff_scores)

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def _evaluate_hardness_logodd(self, pretrain, support, query, support_labels, query_labels):
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=1)
        num_classes = pretrain.num_classes
        n_way = 5

        support_probs = relu(pretrain.classify(support, normalize_prob=False)[:, :num_classes])
        query_probs = relu(pretrain.classify(query, normalize_prob=False)[:, :num_classes])
        # support_probs = pretrain.classify(support, normalize_prob=False)[:, :num_classes]
        # query_probs = pretrain.classify(query, normalize_prob=False)[:, :num_classes]

        w = torch.zeros(n_way, support_probs.shape[1]).cuda()
        for i in range(n_way):
            w[i] = support_probs[support_labels == i].mean(dim=0)
        w = self.normalize(w)

        query_logits = self.normalize(query_probs)
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
        return hardness

    def validate(self, valLoader, lr=None, mode='val'):
        """
        Run one epoch on val-set.
        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param string mode: 'val' or 'train'
        """
        if mode == 'test':
            nEpisode = self.nEpisode
            self.logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))
        elif mode == 'val':
            nEpisode = len(valLoader)
            self.logger.info('\n\nValidation mode: pre-defined {:d} episodes...'.format(nEpisode))
            valLoader = iter(valLoader)
        else:
            raise ValueError('mode is wrong!')

        episodeAccLog = []
        top1 = AverageMeter()

        self.netFeat.eval()
        #self.netSIB.eval() # set train mode, since updating bn helps to estimate better gradient

        if lr is None:
            lr = self.optimizer.param_groups[0]['lr']

        #for batchIdx, data in enumerate(valLoader):
        # nEpisode = 1
        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                # SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
                SupportFeat, QueryFeat = self.pretrain.get_features(SupportTensor), self.pretrain.get_features(QueryTensor)
                SupportFeat, QueryFeat, SupportLabel = \
                        SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)

            # Inductive
            '''
            clsScore = torch.zeros(QueryFeat.shape[1], 5).cuda()
            for i in range(QueryFeat.shape[1]):
                singleScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat[:, i, :].unsqueeze(1), lr)
                clsScore[i] = singleScore[0][0]
            '''

            QueryLabel = QueryLabel.view(-1)

            if self.davg:
                # diff_scores = self.calc_diff_scores(self.pretrain, SupportFeat.squeeze(0), QueryFeat.squeeze(0), SupportLabel.squeeze(0), QueryLabel)  # cosine similarity
                diff_scores = self._evaluate_hardness_logodd(self.pretrain, SupportFeat.squeeze(0), QueryFeat.squeeze(0), SupportLabel.squeeze(0), QueryLabel)  # logodd
            else:
                diff_scores = None
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,), diff_scores=diff_scores)
            top1.update(acc1[0].item(), clsScore.shape[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())

        mean, ci95 = getCi(episodeAccLog)
        msg = 'Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95)
        self.logger.info(msg)
        self.write_output_message(msg)
        return mean, ci95

    def write_output_message(self, message):
        output_file = "results.txt"
        with open(output_file, "a") as f:
            f.write(message)

    def train(self, trainLoader, valLoader, lr=None, coeffGrad=0.0) :
        """
        Run one epoch on train-set.

        :param trainLoader: the dataloader of train-set
        :type trainLoader: class `TrainLoader`
        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param float coeffGrad: deprecated
        """
        bestAcc, ci = self.validate(valLoader, lr, 'test')
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netSIB.train()
        self.netFeat.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)

            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']
                nC, nH, nW = SupportTensor.shape[2:]

                # SupportFeat = self.netFeat(SupportTensor.reshape(-1, nC, nH, nW))
                SupportFeat = self.pretrain.get_features(SupportTensor.reshape(-1, nC, nH, nW))
                SupportFeat = SupportFeat.view(self.batchSize, -1, self.nFeat)

                # QueryFeat = self.netFeat(QueryTensor.reshape(-1, nC, nH, nW))
                QueryFeat = self.pretrain.get_features(QueryTensor.reshape(-1, nC, nH, nW))
                QueryFeat = QueryFeat.view(self.batchSize, -1, self.nFeat)

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']

            self.optimizer.zero_grad()

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)

            # Inductive
            '''
            clsScore = torch.zeros(QueryFeat.shape[1], 5).cuda()
            for i in range(QueryFeat.shape[1]):
                singleScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat[:, i, :].unsqueeze(1), lr)
                clsScore[i] = singleScore[0][0]
            '''
                
            QueryLabel = QueryLabel.view(-1)

            if coeffGrad > 0:
                loss, gradLoss = self.compute_grad_loss(clsScore, QueryLabel)
                loss = loss + gradLoss * coeffGrad
            else:
                loss = self.criterion(clsScore, QueryLabel)

            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.shape[0])
            losses.update(loss.item(), QueryFeat.shape[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            if coeffGrad > 0:
                msg = msg + '| gradLoss: {:.3f}%'.format(gradLoss.item())
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader, lr, 'test')

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)

                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'SIB': self.netSIB.state_dict(),
                                'nbStep': self.nStep,
                                }, os.path.join(self.outDir, 'netSIBBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': self.netFeat.state_dict(),
                            'SIB': self.netSIB.state_dict(),
                            'nbStep': self.nStep,
                            }, os.path.join(self.outDir, 'netSIBLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%, Best Acc {:.3f}'.format(
                        episode, losses.avg, top1.avg, acc, bestAcc)
                self.logger.info(msg)
                self.write_output_message(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history
