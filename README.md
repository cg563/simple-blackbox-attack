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
For targeted attack, add flag --targeted and change --num_iters to 30000.
