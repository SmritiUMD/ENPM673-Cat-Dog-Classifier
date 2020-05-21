import torch
import matplotlib.pyplot as plt
import numpy as np
import os
try:
	import cv2
except:
	import sys
	sys.path.remove(sys.path[1])
	import cv2
import pandas as pd
from torchvision import transforms, datasets, utils
import glob   # loading some train images
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import csv
# from Network import Net
import torch.optim as optim
torch.cuda.is_available()

class CatandDog(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.labels= pd.read_csv(csv_file)
	 
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		image_name = os.path.join(self.root_dir,
							  self.labels.iloc[idx, 0])
		image_name = image_name[:-3]+"jpg"
		image = cv2.imread(image_name)
		label = self.labels.iloc[idx, 1]
		if label==0:
			label_ = np.array([1,0],dtype="f")
		else:
			label_= np.array([0,1],dtype="f")
		sample = {'image': image, 'labels': label_}
		if self.transform:
			sample = self.transform(sample)
		return sample


class Normalize(object):
	def __call__(self, sample):
		image = sample['image']
		label = sample['labels']
		image_copy = np.copy(image)
		image_copy =  image_copy/255.0
		return {'image': image_copy, 'labels': label}

class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		label = sample['labels']
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):  # to check if the output_size is of int type
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = cv2.resize(image, (new_w, new_h))
		return {'image': img, 'labels': label} 


class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		label = sample['labels']
		h, w = image.shape[0:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h,
					  left: left + new_w]

		return {'image': image, 'labels': label}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image = sample['image']
		label = sample['labels']
		# if image has no grayscale color channel, add one
		if(len(image.shape) == 2):
			# add that third color dim
			image = image.reshape(image.shape[0], image.shape[1], 1)
			
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image), 'labels':label}
