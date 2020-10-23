##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for pre-train phase. """
import os

def run_exp(lr=0.1, gamma=0.2, step_size=30, n_clfs=10):
    max_epoch = 80
    shot = 1
    query = 15
    way = 5
    gpu = 2
    base_lr = 0.01
    
    the_command = 'python3 main.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_lr=' + str(lr) \
        + ' --init_weights=' + "/data2/yuezhongqi/Model/mtl/shared/max_acc.pth" \
        + ' --nclfs=' + str(n_clfs) \
        + ' --phase=pretrain_clfs'

    os.system(the_command)

run_exp(lr=0.1, gamma=0.2, step_size=30, n_clfs=1)
