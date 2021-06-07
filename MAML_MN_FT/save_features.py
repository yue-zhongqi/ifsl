import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.wrn_mixup_model import wrn28_10
from model.feat.wrn28 import FEATWRN
from model.sib.networks import SIBWRN
from model.cosine.CosinePretrain import CosinePretrain
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
import simple_shot_models


def save_features(model, data_loader, outfile, params):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    all_paths = np.array([])
    count=0
    with torch.no_grad():
        for i, (x,y, path) in enumerate(data_loader):
            if i%10 == 0:
                print('{:d}/{:d}'.format(i, len(data_loader)))
            x = x.cuda()
            x_var = Variable(x)
            if params.method == "S2M2_R":
                feats, _ = model(x_var)
            elif params.method in ["feat", "sib"]:
                feats = model.forward_feature(x_var)
            else:
                feats = model(x_var)
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
            all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count+feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)
            all_paths = np.concatenate((all_paths, np.array(path)), axis=0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    np.save("img_paths_%s_%s_%s.npy" % (params.dataset, params.model, params.method), all_paths)
    f.close()


def baseline_s2m2_init(params):
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        if params.method == "S2M2_R":
            image_size = 80
        else:
            image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split +'.json' 
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
        else:
            loadfile  = configs.data_dir['emnist'] + split +'.json' 
    else:
        loadfile = configs.data_dir[params.dataset] + split + '.json'

    ###### Temp !!!!!!!!!!!!!!!!!
    if params.dataset == "cross":
        dataset = "miniImagenet"
    else:
        dataset = params.dataset
    ###### Temp !!!!!!!!!!!!!!!!!

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++', 'S2M2_R'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    #    elif params.method in ['baseline', 'baseline++'] :
    #        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 
        if params.dataset == "cross":
            outfile = outfile.replace("miniImagenet", "cross")

    ###### Temp !!!!!!!!!!!!!!!!!
    # outfile = outfile.replace("miniImagenet", "cross")
    ###### Temp !!!!!!!!!!!!!!!!!

    datamgr         = SimpleDataManager(image_size, batch_size = 64)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False, num_workers=12)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            model = backbone.Conv4NP()
        elif params.model == 'Conv6': 
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S': 
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model]( flatten = False )
    elif params.method in ['maml' , 'maml_approx']: 
        raise ValueError('MAML do not support save feature')
    elif params.method == "S2M2_R":
        model = wrn28_10(200)
    else:
        model = model_dict[params.model]()

    print("Using %s" % modelfile)
    
    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    if params.method == "S2M2_R":
        callwrap = False
        if 'module' in state_keys[0]:
            callwrap = True

        if callwrap:
            model = WrappedModel(model) 

        model_dict_load = model.state_dict()
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)
    else:
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
                
        model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return model, data_loader, outfile, params


def remove_module_from_param_name(params_name_str):
    split_str = params_name_str.split(".")[1:]
    params_name_str = ".".join(split_str)
    return params_name_str


def simple_shot_init(params, split):
    model_name = params.model.lower()
    if params.dataset == "cross" or params.dataset == "miniImagenet":
        model_dir = os.path.join(configs.simple_shot_dir, "miniImagenet", model_name, "model_best.pth.tar")
    elif params.dataset == "tiered":
        model_dir = os.path.join(configs.simple_shot_dir, "tiered", model_name, "model_best.pth.tar")
    num_classes = 64
    if params.dataset == "tiered":
        num_classes = 351
    model = simple_shot_models.__dict__[model_name](num_classes=num_classes, remove_linear=True)
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_dir)
    model_dict = model.state_dict()
    model_params = checkpoint['state_dict']
    model_params = {remove_module_from_param_name(k): v for k, v in model_params.items()}
    model_params = {k: v for k, v in model_params.items() if k in model_dict}
    model_dict.update(model_params)
    model.load_state_dict(model_dict)

    tiered_mini = False
    if params.dataset == "cross":
        loadfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == "miniImagenet":
        loadfile = configs.data_dir['miniImagenet'] + split + '.json'
    elif params.dataset == "tiered":
        loadfile = split
        tiered_mini = True
    image_size = 84
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False, num_workers=12, tiered_mini=tiered_mini)

    outfile = '%s/features/%s/%s/%s.hdf5' % (configs.simple_shot_dir, params.dataset, params.model, split)
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return model, data_loader, outfile, params


def feat_init(params, split):
    model_name = params.model.lower()      # wrn
    if params.dataset == "cross" or params.dataset == "miniImagenet":
        model_dir = os.path.join(configs.feat_dir, "miniImagenet", model_name + "_pre.pth")
    model = FEATWRN(64)
    model = model.cuda()

    model_dict = model.state_dict()
    checkpoint = torch.load(model_dir)
    model_dict.update(checkpoint['params'])
    model.load_state_dict(model_dict)
    model.eval()
    
    if params.dataset == "cross":
        loadfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == "miniImagenet":
        loadfile = configs.data_dir['miniImagenet'] + split + '.json'
    image_size = 84
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False, num_workers=0)

    outfile = '%s/features/%s/%s/%s.hdf5' % (configs.feat_dir, params.dataset, params.model, split)
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return model, data_loader, outfile, params


def sib_init(params, split):
    model_name = params.model.lower()
    if params.dataset == "cross" or params.dataset == "miniImagenet":
        model_dir = os.path.join(configs.sib_dir, "miniImagenet", model_name + "_best.pth")
    model = SIBWRN(num_classes=64)
    model = model.cuda()
    model_dict = model.encoder.state_dict()
    checkpoint = torch.load(model_dir)
    model_dict.update(checkpoint)
    model.encoder.load_state_dict(model_dict)
    model.eval()

    if params.dataset == "cross":
        loadfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == "miniImagenet":
        loadfile = configs.data_dir['miniImagenet'] + split + '.json'
    image_size = 80
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False, num_workers=12)

    outfile = '%s/features/%s/%s/%s.hdf5' % (configs.sib_dir, params.dataset, params.model, split)
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return model, data_loader, outfile, params


def cosine_init(params, split):
    model_name = params.model.lower()
    if params.dataset == "cross" or params.dataset == "miniImagenet":
        num_classes = 64
        model_dir = os.path.join(configs.cosine_dir, "miniImagenet", model_name, "max_acc.pth")
    elif params.dataset == "tiered":
        num_classes = 351
        model_dir = os.path.join(configs.cosine_dir, "tiered", model_name, "max_acc.pth")
    if model_name == "resnet10":
        feat_dim = 512
    elif model_name == "wrn":
        feat_dim = 640
    model = CosinePretrain(model_name, num_classes, feat_dim)
    model = model.cuda()
    model_dict = model.encoder.state_dict()
    ckpt = torch.load(model_dir)["params"]
    model_dict.update(ckpt)
    model.encoder.load_state_dict(model_dict)
    model.eval()

    if params.dataset == "cross":
        loadfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == "miniImagenet":
        loadfile = configs.data_dir['miniImagenet'] + split + '.json'
    image_size = 84
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False, num_workers=12)

    outfile = '%s/features/%s/%s/%s.hdf5' % (configs.cosine_dir, params.dataset, params.model, split)
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return model, data_loader, outfile, params

def initialize_and_save(params, split):
    if params.method == "baseline" or params.method == "baseline++" or params.method == "S2M2_R":
        model, data_loader, outfile, params = baseline_s2m2_init(params)
    elif params.method in ["simpleshot", "simpleshotwide"]:
        model, data_loader, outfile, params = simple_shot_init(params, split)
    elif params.method == "feat":
        model, data_loader, outfile, params = feat_init(params, split)
    elif params.method == "sib":
        model, data_loader, outfile, params = sib_init(params, split)
    elif params.method == "cosine":
        model, data_loader, outfile, params = cosine_init(params, split)
    save_features(model, data_loader, outfile, params)


if __name__ == '__main__':
    params = parse_args('save_features')
    # initialize_and_save(params, "base")
    # initialize_and_save(params, "val")
    initialize_and_save(params, "novel")
    

'''
python save_features.py --dataset cross --model wrn --method feat --train_aug
python save_features.py --dataset miniImagenet --model ResNet10 --method simpleshot --train_aug
python save_features.py --dataset miniImagenet --model wideres --method simpleshotwide --train_aug
python save_features.py --dataset miniImagenet --model wrn --method feat --train_aug
python save_features.py --dataset miniImagenet --model wrn --method sib --train_aug
'''
