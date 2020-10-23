# Interventional Few-Shot Learning

This project provides a strong Baseline with WRN28-10 and ResNet-10 backbone for the following Few-Shot Learning methods:

- Fine-tuning
- Matching Networks
- Model-Agnostic Meta-Learning (MAML)
- Meta-Transfer Learning for Few-Shot Learning (MTL)
- Meta-Learning with Latent Embedding Optimization (LEO)
- Synthetic Information Bottleneck (SIB)

This also includes implementation of our **NeurIPS 2020** paper  [Interventional Few-Shot Learning](https://arxiv.org/abs/2009.13000), which proposes IFSL classifier based on intervention P(Y|do(X)) to remove the confounding bias from pre-trained knowledge. Our IFSL classifier is generally applicable to all fine-tuning and meta-learning method, easy to plug in and involves no additional training steps.

The codes are organized into four folders according to methods. The folder *finetune_MN_MAML* contains baseline and IFSL for fine-tuning, Matching Networks and MAML.

## Dependencies

Recommended version:
- Python 3.7.6
- PyTorch 1.4.0

## Preparation

- Download pre-trained backbone in https://github.com/mileyan/simple_shot
- Download mini-ImageNet following https://github.com/hushell/sib_meta_learn
- Tiered-ImageNet download instruction https://github.com/yaoyao-liu/meta-transfer-learning
- Download CUB from official website http://www.vision.caltech.edu/visipedia/CUB-200.html

After downloading the weights and datasets, you can follow the instructions in each folder to modify the code and finish preparation.

## TODO

Apologize in advance for dirty code, which I will clean up gradually.

Before the release of other methods, you can refer to SIB for IFSL implementation (which is really the same across all methods).

- ~~Code release for SIB~~
- ~~Code release for LEO~~
- ~~Code release for MTL~~
- Code release for fine-tuning, MN and MAML (Planned by 25/10/2020)
- Code refactoring
- Improve documentation and optimize project setup procedures

## References

The implementation is based on the following repositories (for correctness of baseline, most of our code is based on the official released code). 

- A Closer Look at Few Shot Learning: https://github.com/wyharveychen/CloserLookFewShot
- Synthetic Information Bottleneck: https://github.com/amzn/xfer/tree/master/synthetic_info_bottleneck
- Meta-Transfer Learning: https://github.com/yaoyao-liu/meta-transfer-learning
- Meta-Learning with Latent Embedding Optimization: https://github.com/deepmind/leo
- SimpleShot: https://github.com/mileyan/simple_shot

## Citation

If you find our work or the code useful, please consider cite our paper using:

```bibtex
@inproceedings{yue2020interventional,
  title={Interventional Few-Shot Learning},
  author={Yue, Zhongqi and Zhang, Hanwang and Sun, Qianru and Hua, Xian-Sheng},
  booktitle= {NeurIPS},
  year={2020}
}
```
