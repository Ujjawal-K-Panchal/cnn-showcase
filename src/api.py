# -*- coding: utf-8 -*-
"""
Title: CNN.

Description: A Convolutional Neural Network.

Created on Sun Jul 10 2022 19:19:01 2022.

@author: Ujjawal .K. Panchal
===

Copyright (c) 2022, Ujjawal Panchal.
All Rights Reserved.
"""
#common libs.
import torch, io, numpy as np
from PIL import Image

#api lib.
from fastapi import FastAPI, File, UploadFile

#import our files.
import main
from model import CNN

#some of our assumptions.
OURIMGSIZE = (32, 32) 


#add
mnist_model = pipeline.load_model(CNN(1), f"MNIST-{main.MODELNAME}")
fashion_model = pipeline.load_model(CNN(1), f"FASHION-{main.MODELNAME}")
app = FastAPI("Our CNN Model")

#classnames.
mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fashion_classes = [
					"T-shirt/top",
					"Trouser",
					"Pullover",
					"Dress",
					"Coat",
					"Sandal",
					"Shirt",
					"Sneaker",
					"Bag",
					"Ankle boot",
]



app.post("/predict/")
def predict_image_class(
	image_file: UploadFile = File(...),
	dataset_name: str = "MNIST",
) -> str:
	"""
	For a file uploaded by user, accept it and print the show class\
	predicted by model.
	---
	Arguments:
		1. image_file: UploadFile = The file of image which you want to predict for.
		2. dataset_name: str (support: MNIST|Fashion) = Name of the type of dataset it is. 
	"""
	#1. load image file to image.
	bytes_content = image_file.read()
	image = Image.open(io.BytesIO(bytes_content)).resize(OURIMGSIZE)
	img_tensor = torch.tensor(image.numpy()[np.newaxis, np.newaxis, :, :]).reshape(0, 3, 1, 2)

	#2. predict the output.
	prediction_soft = softmax(model(img_tensor), dim = 1)
	prediction_hard = prediction_soft.argmax(dim = 1)

	if dataset_name == "MNIST":
		pred_class = mnist_classes[prediction_hard]
	elif dataset_name == "Fashion":
		pred_class = fashion_classes[prediction_hard]
	else:
		raise Exception(f"Dataset '{dataset_name}' is not supported.")
	return pred_class