Torch implementation of semi-supervised Adversarial AutoEncoder on MNIST
===

An implementation of semi-supervised AAE training on MNIST described in the paper 
[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644).


## Requirements

- [Torch7](https://github.com/torch/torch7)
- [penlight](https://github.com/stevedonovan/Penlight)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- [npy4th](https://github.com/htwaijry/npy4th)
- [cutorch](https://github.com/torch/cutorch)
- [cunn](https://github.com/torch/cunn)
- [cudnn](https://github.com/soumith/cudnn.torch)
- Python3 >= 3.5
- Scikit-Learn (for MNIST dataset)

## Usage

First run the preprocessing script that downloads the MNIST dataset and splits
the dataset into labeled training and validation data, and non-labeled training
data.
```
python3 preprocess_mnist.py
```

To train the models, run:
```
th main.lua --ydim <n-classes> --zdim <latent-style-dim> --epochs <n-epochs> --batch <batch-size> [--cuda]
```

Add `--cuda` for GPU training.

Trained models are saved to the `models` directory.
