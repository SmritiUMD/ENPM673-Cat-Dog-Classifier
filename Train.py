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
from VGG16 import Net 
import torch.optim as optim
from Utils import *
torch.cuda.is_available()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Train(data, net):
	images = data['image']
	labels = data['labels']
 
	labels = labels.type(torch.long)
	images = images.type(torch.FloatTensor)
	
	#transferring into cuda
	images=images.to(device)
	labels=labels.to(device)

	output_labels, softmaxOutput = net(images)
	return output_labels, softmaxOutput, labels		

def train_net(net, n_epochs, train_loader, val_loader):
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(params = net.parameters(), lr = 0.0001)
	net.train()
	lossEpoch, accuracyEpoch = [],[]
	for epoch in range(n_epochs):  # loop over the dataset multiple times
		print(epoch)
		running_loss = 0.0
		correct=0.0
		total=0
		loss_values=[]
		accuracy_values=[]
		loss_values_val=[]
		accuracy_values_val=[]
		# train on batches of data, assumes you already have train_loader
		for batch_i, data in enumerate(train_loader):
			
			# forward pass to get outputs
			output_labels, softmaxOutput, labels = Train(data, net)
			
			# loss = criterion(output_labels, labels)
			loss = criterion(output_labels, torch.max(labels, 1)[1])

			# zero the parameter (weight) gradients
			optimizer.zero_grad()
			
			# backward pass to calculate the weight gradients
			loss.backward()

			# update the weights
			optimizer.step()
			running_loss = 0.0
			# print loss statistics
			running_loss += loss.item()
			loss_values.append(running_loss)
			_, predicted = torch.max(output_labels.data, 1)
			total += labels.size(0)

			labels_list = []
			for s in range(0,labels.shape[0]):
			  labels_list.append(labels[s][1])
			labels = torch.tensor(labels_list)
			labels=labels.to(device)
			#print(labels)
			correct += (predicted == labels).sum().item()
		
			accuracy=(100 * correct / total)
			accuracy_values.append(accuracy)
			
			if batch_i % 10 == 9:    # print every 10 batches
				print('Epoch: {}, Batch: {}, Avg. Loss: {}, '.format(epoch + 1, batch_i+1, running_loss))
				print("Accuracy = {}".format(accuracy)) 
		lossEpoch.append(sum(loss_values))
		accuracyEpoch.append(sum(accuracy_values)/len(accuracy_values))
		running_loss = 0
		for batch_i, data in enumerate(val_loader):
			output_labels, softmaxOutput, labels = Train(data, net)
			loss = criterion(output_labels, torch.max(labels, 1)[1])
			running_loss += loss.item()
			loss_values_val.append(running_loss)
			_, predicted = torch.max(output_labels.data, 1)
			total += labels.size(0)

			labels_list = []
			for s in range(0,labels.shape[0]):
			  labels_list.append(labels[s][1])
			labels = torch.tensor(labels_list)
			labels = labels.to(device)
			#print(labels)
			correct += (predicted == labels).sum().item()
		
			accuracy=(100 * correct / total)
			accuracy_values_val.append(accuracy)

		# accuracy_values, loss_values = [],[]
					# print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/x.shape[0]))

	print(accuracyEpoch)
	print(lossEpoch)
	print('Finished Training')
	plt.figure(1)                # the first figure
	plt.subplot(211)             # the first subplot in the first figure
	plt.plot(np.array(lossEpoch), 'r')
	plt.subplot(212)             # the second subplot in the first figure
	plt.plot(np.array(accuracyEpoch), 'b')


# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
									 RandomCrop(224),
									 Normalize(),
									 ToTensor()])

# create the transformed dataset
transformed_dataset = CatandDog(csv_file='train.csv',
									  root_dir='dogs-vs-cats/train/',transform=data_transform)

train_set, val_set = torch.utils.data.random_split(transformed_dataset, [20000, 4999]) 
net = Net().to(device)



batch_size = 32

train_loader = DataLoader(train_set, 
						  batch_size=batch_size,
						  shuffle=True, 
						  num_workers=4)
val_loader = DataLoader(val_set, 
						  batch_size=batch_size,
						  shuffle=False, 
						  num_workers=4)


criterion = nn.CrossEntropyLoss()

train_net(net, 10, train_loader, val_loader)

path = "../CheckPoint/model"
torch.save(net.state_dict(), path)
# net.load_state_dict(torch.load(path))
# net.eval()

def test():
	net.eval()
	with open('test2.csv', mode='w') as file:
		for i, (data) in enumerate(train_loader):
			images = data['image']
			# print(labels)

			images = images.type(torch.FloatTensor)
			# #transferring into cuda
			images=images.to(device)

			# Predict classes using images from the test set
			outputs = model['image']
			_, prediction = torch.max(outputs.data, 1)
			writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow([images,prediction])

	return prediction

