import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n = 8, dropout_value = 0.1):
        super(Net, self).__init__()
        # # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 28 x 28 x 1, output size: 26 x 26 x n, receptive field: 1 + (3-1) * 1 = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 26 x 26 x n, output size: 24 x 24 x n*2, receptive field: 3 + (3-1) * 1 = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) #input size: 24 x 24 x n*2, output size: 12 x 12 x n*2, receptive field: 5 + (2-1) * 1 = 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 12 x 12 x n*2, output size: 12 x 12 x n, receptive field: 6 + (1-1)*2 = 6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 12 x 12 x n, output size: 10 x 10 x n*2, receptive field: 6 + (3-1) * 2 = 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 10 x 10 x n*2, output size: 8 x 8 x n*2, receptive field: 10 + (3-1) * 2 = 14

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*2, output size: 6 x 6 x n*2, receptive field: 14 + (3-1) * 2 = 18

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7) # 7>> 9...
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 6 x 6 x n*2, output size: 1 x 1 x n*2, receptive field: 18

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        ) #input size: 1 x 1 x n*2, output size: 1 x 1 x 10, receptive field: 18 + (1-1) * 2 =18



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.pool1(x)
        x = self.convblock3(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
 
        x = self.gap(x)
        x = self.convblock7(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)
      
