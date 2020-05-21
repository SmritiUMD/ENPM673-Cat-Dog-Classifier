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
from Network import Net
# from VGG16 import Net 
import torch.optim as optim
# from Utils import *
torch.cuda.is_available()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


	
net = Net().to(device)
# path = "../CheckPoint/model"
# net.load_state_dict(torch.load(path))
# net.eval()
# root_dir='dogs-vs-cats/train/'
TestDir = "dogs-vs-cats/test1"
imageNames = glob.glob(TestDir+"/*")
for name in imageNames:
	img = cv2.imread(name)
	image = cv2.resize(img, (224,224))
	image = image.transpose((2, 0, 1))
	image = np.float32(np.expand_dims(image,0))
	ImgTensor = torch.from_numpy(image)
	logits, softmaxoutput = net(ImgTensor)
	print("\nOutput for image ",name," is : ",torch.argmax(logits), logits)
	# print(logits)


