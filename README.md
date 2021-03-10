This repository contains code for the ICML 2019 paper:

Chuan Guo, Jacob R. Gardner, Yurong You, Andrew G. Wilson, Kilian Q. Weinberger. Simple Black-box Adversarial Attacks.
https://arxiv.org/abs/1905.07121

Our code uses PyTorch (pytorch >= 0.4.1, torchvision >= 0.2.1) with CUDA 9.0 and Python 3.5. The script run_simba.py contains code to run SimBA and SimBA-DCT with various options.

To run SimBA (pixel attack):
```
python run_simba.py --data_root <imagenet_root> --num_iters 10000 --pixel_attack  --freq_dims 224
```
To run SimBA-DCT (low frequency attack):
```
python run_simba.py --data_root <imagenet_root> --num_iters 10000 --freq_dims 28 --order strided --stride 7
```
For targeted attack, add flag ```--targeted``` and change ```--num_iters``` to 30000.

For the Inception-v3 model, we used ```--freq_dims 38``` and ```--stride 9``` due to the larger input size.

**Update 2020/01/09**: Due to changes in the underlying Google Cloud Vision models, our attack no longer works against them.

**Update 2020/06/22**: Added L_inf bounded SimBA-DCT attack. To run with L_inf bound 0.05:
```
python run_simba.py --data_root <imagenet_root> --num_iters 10000 --freq_dims 224 --order rand --linf_bound 0.05
```

**Update 2021/03/10**: Added code for running SimBA on CIFAR-10 using models from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). To run targeted attack against a trained ResNet-18 model:
```
python run_simba_cifar.py --data_root <cifar_root> --model_ckpt <model_checkpoint> --model ResNet18 --targeted
```