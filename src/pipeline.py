# -*- coding: utf-8 -*-
"""
Title: Main training the model. 
	
Created on Mon Jul 16 18:58:29 2022

@author: Ujjawal.K.Panchal
"""
import argparse
import torch

from model import CNN

#device mapper.

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



#unit test.
if __name__ == "__main__":
    print(f"unit tests for the pipeline module.")
    
    #1. loadmodels.
    model = load_model(CNN, c = 1)
    print(f"load_model() works fine.")

