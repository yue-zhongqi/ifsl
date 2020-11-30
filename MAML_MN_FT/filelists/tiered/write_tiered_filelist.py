import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import torch.utils.data as data
from torchvision import datasets, transforms


cwd = os.getcwd() 
data_path = "/data2/yuezhongqi/Dataset/tiered"
savedir = './'
dataset_list = ['base','val','novel']

for dataset in dataset_list:
    if dataset == "base":
        split = "train"
    elif dataset == "val":
        split = "val"
    elif dataset == "novel":
        split = "test"
    split_path = os.path.join(data_path, split)
    folder_list = listdir(split_path)
    file_list = []
    label_list = []
    l = -1
    for folder in folder_list:
        l += 1
        folder_path = os.path.join(split_path, folder)
        imgs = listdir(folder_path)
        for img in imgs:
            file_list.append(os.path.join(folder_path, img))
            label_list.append(l)

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
