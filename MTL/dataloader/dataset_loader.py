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
""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, dataset="miniImagenet", train_aug=False, require_path=False):
        if dataset == "miniImagenet":
            dataset_dir = args.mini_dataset_dir
        elif dataset == "tiered":
            dataset_dir = args.tiered_dataset_dir
        elif dataset == "cross":
            dataset_dir = args.cross_dataset_dir
        # Set the path according to train, val and test
        if dataset == "cross":
            # Cross uses the entire dataset for testing
            THE_PATH = dataset_dir
            label_list = os.listdir(dataset_dir)
        else:
            if setname=='train':
                if dataset == "miniImagenet":
                    train_folder = "train_train"
                else:
                    train_folder = "train"
                THE_PATH = osp.join(dataset_dir, train_folder)
                label_list = os.listdir(THE_PATH)
            elif setname=='test':
                THE_PATH = osp.join(dataset_dir, 'test')
                label_list = os.listdir(THE_PATH)
            elif setname=='val':
                THE_PATH = osp.join(dataset_dir, 'val')
                label_list = os.listdir(THE_PATH)
            else:
                raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.require_path = require_path

        # Transformation
        if dataset == "miniImagenet":
            mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
            std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        elif dataset == "tiered":
            mean = [x/255.0 for x in [125.3,  123.0, 113.9]]
            std = [x/255.0 for x in [63.0,  62.1,  66.7]]
        elif dataset == "cross":
            mean = [0.4859, 0.4996, 0.4318]
            std = [0.1822, 0.1812, 0.1932]

        if train_aug:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if not self.require_path:
            return image, label
        else:
            return image, label, path
