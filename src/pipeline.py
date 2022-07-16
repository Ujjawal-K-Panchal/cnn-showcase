# -*- coding: utf-8 -*-
"""
Title: Main training the model. 
	
Created on Mon Jul 16 18:58:29 2022

@author: Ujjawal.K.Panchal
"""

import argparse, time
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms

from tqdm import tqdm
from model import CNN
from typing import Union

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
    transform = transforms.ToTensor(),
    download: bool = True,
    **kwargs
) -> datasets:
    """
        load_dataset() helps the user to load any dataset from supported ones.
    """
    assert dname in SUPPORTED_DATASETS, f"<!>: DATASET: `{dname}` not supported." 
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

def get_dataLoader(
    dset,
    batch_size: int = 32,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """
    Get an iterable form for the dataset.
    ---
    Args:
        1. batch_size: int (default = 32) = size of each batch coming out of the set.
        2. shuffle: bool (default = True) = if to or not to shuffle the dataset when creating batches.
    """
    dLoader = torch.utils.data.DataLoader(
                        dset,
                        batch_size = batch_size,
                        shuffle = shuffle
                )
    return dLoader

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 1,
    lr: float = 1E-3,
    device: Union[str, torch.device] = "cpu",
    optimizer: torch.optim = torch.optim.Adam,
    loss_fn: torch.nn = torch.nn.CrossEntropyLoss(),

) -> torch.nn.Module:
    """
    Train a network on a given dataset iterable.
    ---
    Args:
        1. model: torch.nn.Module (req) = model which to train.
        2. train_loader: torch.utils.data.DataLoader (req) = train set iterable.
        3. test_loader: torch.utils.data.DataLoader (req) = test set iterable.
    """
    model.to(device)

    model.train()
    optimizer = optimizer(model.parameters(), lr = lr)
    softmax = F.softmax
    correct = 0
    total = 0
    total_loss = 0 

    #0. run n epochs.
    for epoch in range(epochs):
        with tqdm(train_loader, unit = "batch") as train_epoch:
            #1. set some bat stats.
            train_epoch.set_description(f"E {epoch}, loss: {total_loss:.2f}, train acc: {correct * 100 /total if total else total:.2f}%")
            total_loss = 0
            correct = 0
            total = 0
            for data, target in train_epoch:
                #2. get data and predict.
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = softmax(model(data), dim = 1)
                preds = output.argmax(dim = 1)

                
                #3. calculate loss & backpropogate gradients.
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                #4. some intermediate stat collection.
                correct += (preds == target).sum().item()
                total += len(preds)
                total_loss += loss.item()
    return model

#unit test.
if __name__ == "__main__":
    print(f"unit tests for the pipeline module.")
    
    #1. load_models.
    model = load_model(CNN, c = 1)
    print(f"load_model() works fine.")

    #2. load_datasets.
    train_set, test_set = load_dataset(
        SUPPORTED_DATASETS[0],

    )
    print(f"load_dataset() works fine.")

    #3. get_dataLoader.
    loader = get_dataLoader(test_set, batch_size=512)
    print(f"get_dataLoader() works fine.")

    #4. train.
    model = train(model, loader)
    print(f"train() works fine.")
