# -*- coding: utf-8 -*-
"""
Title: Main training the model. 
	
Created on Mon Jul 16 18:58:29 2022

@author: Ujjawal.K.Panchal
"""
#imports.
import argparse, torch

from torchvision import transforms

import model, pipeline

#static vars.
LR = 1E-4
BS = 128
EPOCHS = 100
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda:0"
NUMCHANNELS = 1
DATASET = "MNIST"


if __name__ == "__main__":
	#1. load dataset.
	train_set, test_set = pipeline.load_dataset(
									DATASET,
									transform = transforms.ToTensor(),
							)
	print(f"Dataset: {type(train_set)=}")
	print(f"Dataset sample: {type(train_set[0][0])=}, {train_set[0][0].size()=}")

	#2. make dataset iterable, shard as batches.
	train_loader = pipeline.get_dataLoader(
						dset = train_set,
						batch_size = BS,
						shuffle = True
					)
	test_loader = pipeline.get_dataLoader(
						dset = test_set,
						batch_size = BS * 2, #Note: this is *2 to speedup testing. test bs doesn't matter.
						shuffle = False
					)

	#2. load our model.
	model = pipeline.load_model(model.CNN, DEVICE, NUMCHANNELS)






