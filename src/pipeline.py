# -*- coding: utf-8 -*-
"""
Title: Main training the model. 
	
Created on Mon Jul 16 18:58:29 2022

@author: Ujjawal.K.Panchal
"""

import argparse
import torch
import torchvision.datasets as datasets

from functools import partial
from model import CNN

#static_vars.
SUPPORTED_DATASETS = ["MNIST", "Fashion"]

dataset_loaders = {
    "MNIST" :  datasets.MNIST,
    "Fashion": datasets.FashionMNIST,
}

#load model.
def load_model(
    model_class: torch.nn.Module = CNN,
    device: str = "cpu",
    *args, **kwargs
):
    """
    load_model() helps the user to load any model from any class.

    ---
    Arguments:
        1. model_class: torch.nn.Module (default CNN) = any modelclass which is `torch.nn.Module`.
        2. device: str (default cpu) = any device str available on system.
        3. *args: any extra positional arguments; passed to `model_class`.
        4. **args: any extra keyword arguments; passed to `model_class`.
    """
    model = model_class(*args, **kwargs).to(device)
    return model

#load dataset.
def load_dataset(
    dname: str = SUPPORTED_DATASETS[0],
    root: str = "./data",
    transform = None,
    download: bool = True,
    **kwargs
):
    """
        load_dataset() helps the user to load any dataset from supported ones.
    """
    train_set = dataset_loaders[dname](
                        root = root,
                        train = True,
                        transform = transform,
                        download = download,
                        **kwargs
                    
                )
    test_set = dataset_loaders[dname](
                        root = root,
                        train = False,
                        transform = transform,
                        download = download,
                        **kwargs
                )
    return  train_set, test_set



#unit test.
if __name__ == "__main__":
    print(f"unit tests for the pipeline module.")
    
    #1. load_models.
    model = load_model(CNN, c = 1)
    print(f"load_model() works fine.")

    #2. load_datasets.
    train_set, test_set = load_dataset(
        SUPPORTED_DATASETS[0]
    )
    print(f"load_dataset() works fine.")
