# -*- coding: utf-8 -*-
"""
Title: Main training the model. 
	
Created on Mon Jul 16 18:58:29 2022

@author: Ujjawal.K.Panchal
"""
#imports.
import argparse, torch
import model, pipeline

#static vars.
LR = 1E-4
BS = 128
EPOCHS = 100
DEVICE = "CPU" if not torch.cuda.is_available() else "cuda:0"
NUMCHANNELS = 1
DATASET = "MNIST"


if __name__ == "__main__":
	#1. load dataset.
	train_set, test_set = pipeline.load_dataset(DATASET)
	
	#2. load our model.
	model = pipeline.load_model(model.CNN, DEVICE, NUMCHANNELS)






