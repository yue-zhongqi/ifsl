# IFSL + Matching Networks, MAML

This project is based on the official code base of the paper [A Closer Look At Few-Shot Classification](https://arxiv.org/abs/1904.04232).  IFSL implementations are added in the *methods* folder. The folder pretrain contains pre-saved class-wise feature means used for class-wise adjustment. The folder *tests* contains the running configurations.

## Dependencies

Recommended version:

- Python 3.7.6
- PyTorch 1.4.0

## Preparation

- Download pre-trained backbone in https://github.com/mileyan/simple_shot
- Download mini-ImageNet following https://github.com/wyharveychen/CloserLookFewShot
- Tiered-ImageNet download instruction https://github.com/yaoyao-liu/meta-transfer-learning

Once the datasets are downloaded, go to *filelists/miniImagenet/write_miniImagenet_filelist.py* and *filelists/tiered/write_tiered_filelist.py*. Change *data_path* to the dataset location. Then run the two scripts.

Go to *configs.py.* Change *save_dir* to desired save path for trained models. Change *simple_shot_dir* to the directory where pre-trained weight is stored. Change *tiered_dir* to tiered-ImageNet directory.

## Train and Test

The file tests/MetaTrain.py contains all the configurations to run MAML/Matching Networks Baseline/IFSL with either ResNet10 or WRN-28-10. Run main.py for meta-training followed by meta-testing. Two examples are given below:

```
python main.py --method metatrain --train_aug --test maml5_resnet  # MAML 5 shot miniImageNet with ResNet10
python main.py --method metatrain --train_aug --test maml5_ifsl_resnet_tiered  # MAML 5 shot tieredImageNet with ResNet10
```