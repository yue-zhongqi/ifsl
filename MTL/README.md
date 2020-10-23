# MTL + IFSL

This project is based on the official code base of the paper [Meta Transfer Learning for Few-Shot Learning](https://arxiv.org/abs/1812.02391).  IFSL implementations are added in models folder. The folder pretrain contains pre-saved class-wise feature means used for class-wise adjustment. The folder configs contains the running configurations.

## Dependencies

Recommended version:

- Python 3.7.6
- PyTorch 1.4.0

## Preparation

- Download pre-trained backbone in https://github.com/mileyan/simple_shot
- Download mini-ImageNet following https://github.com/hushell/sib_meta_learn
- Download CUB dataset from  <http://www.vision.caltech.edu/visipedia/CUB-200.html>
- Tiered-ImageNet download instruction https://github.com/yaoyao-liu/meta-transfer-learning

Once the pre-trained model and datasets are downloaded, modify the miniImageNet, tieredImageNet, CUB dataset location in main.py. Additionally, change param.init_weights in each configuration to where you store the pre-trained model.

## Running Experiments

Meta Training:

```
python main.py --config=mini_5_resnet_baseline --gpu=0	 # MTL resnet miniImageNet 5 shot on GPU0
python main.py --config=mini_5_resnet_d --gpu=0	 # MTL+IFSL resnet miniImageNet 5 shot on GPU0
python main.py --config=mini_5_wrn_d --gpu=0,1,2	 # MTL+IFSL WRN miniImageNet 5 shot; 3GPUs are needed
```

Meta Testing:

```
python main.py --config=mini_5_resnet_baseline --gpu=0 --phase=meta_eval
python main.py --config=mini_5_resnet_baseline --gpu=0 --phase=meta_eval --cross=True # Domain generalization experiment miniImageNet->CUB
```