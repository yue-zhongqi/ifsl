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

import torch
import torch.nn as nn
import random
import itertools
import json
import os

from algorithm import Algorithm
from networks import get_featnet
from sib import ClassifierSIB
from deconfound.DSIB import DeconfoundedSIB
from dataset import dataset_setting
from dataloader import BatchSampler, ValLoader, EpisodeSampler
from utils.config import get_config
from utils.utils import get_logger, set_random_seed
from PretrainedModel import PretrainedModel

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Params:
    def __init__(self):
        self.dummy = True

#############################################################################################
## Read hyper-parameters
args = get_config()

# Setup logging to file and stdout
logger = get_logger(args.logDir, args.expName)

logger.info(args)

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.gpu != '':
    args.cuda = True
device = torch.device('cuda' if args.cuda else 'cpu')

#############################################################################################
## Datasets
trainTransform, valTransform, inputW, inputH, \
            trainDir, valDir, testDir, episodeJson, nbCls = \
            dataset_setting(args.dataset, args.nSupport, args.image_size)

trainLoader = BatchSampler(imgDir = trainDir,
                           nClsEpisode = args.nClsEpisode,
                           nSupport = args.nSupport,
                           nQuery = args.nQuery,
                           transform = trainTransform,
                           useGPU = args.cuda,
                           inputW = inputW,
                           inputH = inputH,
                           batchSize = args.batchSize)

'''
valLoader = ValLoader(episodeJson,
                      valDir,
                      inputW,
                      inputH,
                      valTransform,
                      args.cuda)
'''
valLoader = EpisodeSampler(imgDir = valDir,
                            nClsEpisode = args.nClsEpisode,
                            nSupport = args.nSupport,
                            nQuery = args.nQuery,
                            transform = valTransform,
                            useGPU = args.cuda,
                            inputW = inputW,
                            inputH = inputH)

testLoader = EpisodeSampler(imgDir = testDir,
                            nClsEpisode = args.nClsEpisode,
                            nSupport = args.nSupport,
                            nQuery = args.nQuery,
                            transform = valTransform,
                            useGPU = args.cuda,
                            inputW = inputW,
                            inputH = inputH)


#############################################################################################
## Networks
netFeat, args.nFeat = get_featnet(args.architecture, inputW, inputH)
if args.p_model == "ResNet10":
    args.nFeat = 512
netFeat = netFeat.to(device)
# param = torch.load(args.resumeFeatPth)
# netFeat.load_state_dict(param)

params = Params()
params.dataset = args.p_dataset
params.method = args.p_method
params.model = args.p_model
params.num_classes = nbCls
pretrain = PretrainedModel(params)
pretrain.netFeat = netFeat

if args.deconfounding:
    netSIB = DeconfoundedSIB(args.nClsEpisode, pretrain, args.n_splits, args.is_cosine_feature, args.d_feature,
                             args.fusion, args.sum_log, args.lr, args.classifier, args.use_counterfactual, args.x_zero,
                             args.nStep, nbCls, 1, args.logit_fusion, args.use_x_only, args.feat_dim,
                             args.preprocess_after_split, args.preprocess_before_split, args.normalize_before_center,
                             args.normalize_d, args.normalize_ed).to(device)
    criterion = nn.NLLLoss()
    print("Using deconfounding")
else:
    netSIB = ClassifierSIB(args.nClsEpisode, args.nFeat, args.nStep)
    netSIB = netSIB.to(device)
    criterion = nn.CrossEntropyLoss()
    print("Running Baseline")

## Optimizer
'''
optimizer = torch.optim.SGD(itertools.chain(*[netSIB.parameters(),]),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weightDecay,
                            nesterov=True)
'''


optimizer = torch.optim.Adam(itertools.chain(*[netSIB.parameters(),]),
                             args.lr,
                             weight_decay=args.weightDecay)
                            
## Algorithm class
alg = Algorithm(args, logger, netFeat, netSIB, optimizer, criterion, pretrain)


#############################################################################################
## Training
if not args.test:
    bestAcc, lastAcc, history = alg.train(trainLoader, valLoader, coeffGrad=args.coeffGrad)

    ## Finish training!!!
    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBBest.pth'),
                            os.path.join(args.outDir, 'netSIBBest{:.3f}.pth'.format(bestAcc)))
    logger.info(msg)
    os.system(msg)

    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBLast.pth'),
                            os.path.join(args.outDir, 'netSIBLast{:.3f}.pth'.format(lastAcc)))
    logger.info(msg)
    os.system(msg)

    with open(os.path.join(args.outDir, 'history.json'), 'w') as f :
        json.dump(history, f)

    msg = 'mv {} {}'.format(args.outDir, '{}_{:.3f}'.format(args.outDir, bestAcc))
    logger.info(msg)
    os.system(msg)


#############################################################################################
## Testing
logger.info('Testing model {}...'.format(args.ckptPth if args.test else 'LAST'))
mean, ci95 = alg.validate(testLoader, mode='test')

if not args.test:
    logger.info('Testing model BEST...')
    alg.load_ckpt(os.path.join('{}_{:.3f}'.format(args.outDir, bestAcc),
                               'netSIBBest{:.3f}.pth'.format(bestAcc)))
    mean, ci95 = alg.validate(testLoader, mode='test')
