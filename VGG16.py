import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
     	# Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 2)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 2)
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 2)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 2)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 2)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 2)
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 8192, out_features = 4096)
        self.fc2 = nn.Linear(in_features = 4096,    out_features = 4096)
        self.fc3 = nn.Linear(in_features = 4096,    out_features = 2) 
        # the output 2 - 2 classes
        # Dropouts
        self.drop = nn.Dropout(p = 0.5)

        #Batch normalization
        self.bn1 = nn.BatchNorm1d(4096)  
        self.bn2 = nn.BatchNorm1d(2)      
        # self.softmaxLayer = nn.Softmax()
    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        #print("First size: ", x.shape)

        # Second - Convolution + Activation + Convolution+ Activation+ Pooling 
        x = self.conv4(F.relu(self.conv3(x)))
        x = self.pool(F.relu(x))
        #print("Second size: ", x.shape)

        # Third - Convolution + Activation + Convolution+ Activation+ Pooling 
        x = self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x= self.pool(F.relu(x))
        #print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Convolution + Activation + Convolution + Activation + Pooling
        x = self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))
        x= self.drop(self.pool(F.relu(x)))

        #print("Forth size: ", x.shape)

        # Fifth- Convolution + Activation + Convolution + Activation + Convolution + Activation + Pooling
        x = self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))
        x= self.drop(self.pool(F.relu(x)))
        #print("Forth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        # First - Dense + batch normalization+  Activation + Dropout
        x = self.bn1(self.fc1(x))
        x = self.drop(F.relu(x))
        #print("First dense size: ", x.shape)

        # Second - Dense + batch normalization+ Activation + Dropout
        x = self.bn1(self.fc2(x))
        x= self.drop(F.relu(x))
        #print("Second dense size: ", x.shape)

        # Final Dense Layer + batch normalization
        x = self.bn2(self.fc3(x))
        print("Final dense size: ", x.shape)
        output = softmax(x)
        # print(output.shape)
        return x, output