import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Conv2Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(Conv2Block, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = in_channels, 
			out_channels = out_channels, kernel_size = kernel_size)
		self.bn1 = nn.BatchNorm2d(out_channels)
		
		self.conv2 = nn.Conv2d(in_channels = out_channels, 
			out_channels = out_channels, kernel_size = kernel_size)
		self.bn2 = nn.BatchNorm2d(out_channels)
		
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

	def forward(self, x):
		out = x
		out = F.relu(self.bn1(self.conv1(out)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.pool(out)
		return out 

class Conv3Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(Conv3Block, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = in_channels, 
			out_channels = out_channels, kernel_size = kernel_size)
		self.bn1 = nn.BatchNorm2d(out_channels)
		
		self.conv2 = nn.Conv2d(in_channels = out_channels, 
			out_channels = out_channels, kernel_size = kernel_size)
		self.bn2 = nn.BatchNorm2d(out_channels)
		
		self.conv3 = nn.Conv2d(in_channels = out_channels, 
			out_channels = out_channels, kernel_size = kernel_size)
		self.bn3 = nn.BatchNorm2d(out_channels)

		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

	def forward(self, x):
		out = x
		out = F.relu(self.bn1(self.conv1(out)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.relu(self.bn3(self.conv3(out)))
		out = self.pool(out)
		return out 

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()	   
		# Covolutional Layers
		self.block1 = Conv2Block(3, 64, 5)
		self.block2 = Conv2Block(64, 128, 3)
		self.block3 = Conv3Block(128, 256, 3)
		self.block4 = Conv3Block(256, 512, 3)
		self.block5 = Conv3Block(512, 512, 3)
		
		# Maxpooling Layer
		
		# Fully Connected Layers
		self.fc1 = nn.Linear(in_features = 512, out_features = 1024)
		self.fc2 = nn.Linear(in_features = 1024,    out_features = 512)
		self.fc3 = nn.Linear(in_features = 512,    out_features = 2) 
		# the output 2 - 2 classes
		# Dropouts
		self.drop = nn.Dropout(p = 0.5)
		#Batch normalization

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		# Flattening the layer
		x = x.view(x.size(0), -1)
		# print("Flatten size: ", x.shape)

		# First - Dense + batch normalization+  Activation + Dropout
		x = self.fc1(x)
		#print("First dense size: ", x.shape)

		# Second - Dense + batch normalization+ Activation + Dropout
		x = self.fc2(x)
		x = F.relu(x)

		# Final Dense Layer + batch normalization
		prLogits = self.fc3(x)
		#print("Final dense size: ", x.shape)
		prSoftMax = F.log_softmax(x, dim=1)
		return prLogits, prSoftMax