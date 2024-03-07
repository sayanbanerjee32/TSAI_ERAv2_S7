# import required packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,n = 8):
        super(Net, self).__init__()
        # Convolusion Block - Inital
        #input size: 28 x 28 x 1, output size: 28 x 28 x n, receptive field: 3
        self.convi = nn.Conv2d(1, n, 3, padding=1, bias = False)
        self.bni = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 28 x 28 x n, output size: 28 x 28 x n*2, receptive field: 3 + (3-1) * 1 = 5
        self.conv1 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(n*2)
        #input size: 28 x 28 x n*2, output size: 28 x 28 x n*4, receptive field: 5 + (3-1) * 1 = 7
        self.conv2 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(n*4)

        # Transition block - reduction in channel size and number
        #input size: 28 x 28 x n*4, output size: 14 x 14 x n*4, receptive field: 7 + (3-1) * 1 = 8
        self.pool1 = nn.MaxPool2d(2, 2)
        #input size: 14 x 14 x n*4, output size: 14 x 14 x n, receptive field: 8 + (1-1)*2 = 8
        self.antman1 = nn.Conv2d(n*4, n, kernel_size=1,bias = False)
        self.bna1 = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 14 x 14 x n, output size: 14 x 14 x n*2, receptive field: 8 + (3-1) * 2 = 12
        self.conv3 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(n*2)
        #input size: 14 x 14 x n*2, output size: 14 x 14 x n*4, receptive field: 12 + (3-1) * 2 = 16
        self.conv4 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(n*4)

        # Transition block - reduction in channel size and number
        #input size: 14 x 14 x n*4, output size: 7 x 7 x n*4, receptive field: 16 + (2-1) * 2 =18
        self.pool2 = nn.MaxPool2d(2, 2)
        #input size: 7 x 7 x n*4, output size: 7 x 7 x n, receptive field: 18 + (1-1) * 4 =18
        self.antman2 = nn.Conv2d(n*4, n, kernel_size=1, bias = False)
        self.bna2 = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 7 x 7 x n, output size: 7 x 7 x n*2, receptive field: 18 + (3-1) * 4 = 26
        self.conv5 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn5 = nn.BatchNorm2d(n*2)
        #input size: 7 x 7 x n*2, output size: 7 x 7 x n*4, receptive field: 26 + (3-1) * 4 = 34
        self.conv6 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn6 = nn.BatchNorm2d(n*4)

        # Transition block - - reduction in channel size
        # and aligning number of channels to number of prediction classes
        #input size: 7 x 7 x n*4, output size: 3 x 3 x n*4, receptive field: 34 + (2-1) * 4 = 38
        self.pool3 = nn.MaxPool2d(2, 2)
        #input size: 3 x 3 x n*4, output size: 3 x 3 x 10, receptive field: 38 + (1-1) * 8 = 38
        self.antman3 = nn.Conv2d(n*4, 10, kernel_size=1, bias = False)

    def forward(self, x):
        x = self.bni(F.relu(self.convi(x))) #28 x 28

        x = self.bn1(F.relu(self.conv1(x))) #28 x 28
        x = self.bn2(F.relu(self.conv2(x))) #28 x 28
        x = self.bna1(self.antman1(self.pool1(x))) #14 x 14
        # dropout of 0.25 was not allowing model to train to the required level
        # thus dropout is set to 0.1
        x = F.dropout(x, 0.10) #14 x 14

        x = self.bn3(F.relu(self.conv3(x))) #14 x 14
        x = self.bn4(F.relu(self.conv4(x))) #14 x 14
        x = self.bna2(self.antman2(self.pool2(x))) #7 x 7
        # dropout of 0.25 was not allowing model to train to the required level
        # thus dropout is set to 0.1
        x = F.dropout(x, 0.10) #7 x 7

        x = self.bn5(F.relu(self.conv5(x))) #7 x 7
        x = self.bn6(F.relu(self.conv6(x))) #7 x 7
        x = self.antman3(self.pool3(x)) #3 x 3

        # Global average pooling instead of FC layer
        #input size: 3 x 3 x 10, output size: 1 x 1 x 10 > len of 10
        x = F.avg_pool2d(x, kernel_size = 3).squeeze()

        # x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        # x = x.view(-1, 10)
        return F.log_softmax(x)
