# SIB + IFSL

This project is based on the official code base of the paper [Empirical Bayes Transductive Meta-Learning with Synthetic Gradients](https://openreview.net/forum?id=Hkg-xgrYvH).  Codes added for IFSL include deconfound folder (SIB+IFSL implementation), PretrainedModel.py (implementation of pre-trained knowledge) and pretrain folder (saved mean features for class-wise and combined adjustment), simple_shot_models folder (backbone implementation).

## Dependencies

Recommended version:
- Python 3.7.6
- PyTorch 1.4.0

## Preparation

- Download pre-trained backbone in https://github.com/mileyan/simple_shot
- Download mini-ImageNet following https://github.com/hushell/sib_meta_learn
- Tiered-ImageNet download instruction https://github.com/yaoyao-liu/meta-transfer-learning

Once the pre-trained model and datasets are downloaded, modify dataset.py trainDir, valDir and testDir to reflect where you store your dataset and modify dfsl_configs.py to specify where you want to store the trained model and where you store your pre-trained weights.

## Running Experiments

The following command will meta-train a model followed by meta-test.

```
python main.py --config config/FILE_NAME.yaml --gpu GPU
```

For example, to run ResNet-10 5 shot baseline on tieredImageNet with GPU0, set FILE_NAME=tieredres_5_baseline.yaml and GPU=0. Similarly you can run IFSL by using FILE_NAME=tieredres_5_ifsl.yaml.