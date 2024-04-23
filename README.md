# Local Descriptor Development Framework Based on Python
This is the official PyTorch implementation of the paper *SDNet: Spatial Adversarial Perturbation Local Descriptor Learned with the Dynamic Probabilistic Weighting Loss*

# Dependence
Maybe some extra modules are needed. I'm not sure.
- Python 3.11
- PyTorch 2.1
- OpenCV-Python 4.8.1
- Scipy

# Usage

# Dataset
Both the PhotoTourim datasets and HPatches datasets will be automatically downloaded when
the training / testing script runs. By default they will be downloaded to `./data/`. We use
*dot * to separate the dataset name and the split name. PhotoTourism(Brown) datasets are
named as `"brown.liberty"`, `"brown.yosemite"` and `"brown.notredame"` respectively. We
use the full HPatches dataset during training, which is named as `"HP.full"`.

# Training
Use the following command to train a hynet-based model with the `liberty` sequence of PhotoTourism.
```sh
train_sp.py --num_epoch 200 --test_every_epoch --save_every_epoch --bs 1024 --optim adam --lr 0.001 --lr_policy cos --arch_type hynet --drop_rate 0.3 --out_dim 128 --loss_type sdnet --margin 0.45 --alpha_init 0 --alpha_moment 0.995 --quantile 0.0 1.0 --lambda_clean 1 --lambda_adv 1 --adv_step 0.1 --adv_iter 3 --train_data brown.liberty --test_data brown.yosemite brown.notredame HP.a --patch_size 32
```
For hardnet-based model:
```sh
CUDA_VISIBLE_DEVICES=0 python train_sp.py --num_epoch 200 --test_every_epoch --save_every_epoch --bs 1024 --optim adam --lr 0.001 --lr_policy cos --arch_type l2net --drop_rate 0.1 --out_dim 128 --loss_type sdnet --margin 0.45 --alpha_init 0 --alpha_moment 0.995 --quantile 0.0 1.0 --lambda_clean 1 --lambda_adv 1 --adv_step 0.1 --adv_iter 3 --train_data brown.liberty --test_data brown.yosemite brown.notredame HP.a --patch_size 32
```

# Testing
Use the following command to test a real - valued descriptor model:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_dir ./pretrained/SDNet_LIB.state_dict --test_data brown.liberty brown.yosemite brown.notredame HP.a --arch_type hynet
```

# Acknowledgement
This code borrows heavily from dynamic-soft-margin-pytorch. We thank Zhang, Linguang and Rusinkiewicz, Szymon for releasing their code.