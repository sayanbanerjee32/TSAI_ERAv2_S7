class Net(nn.Module):
    def __init__(self, n = 4):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n),
            nn.ReLU()
        ) #input size: 28 x 28 x 1, output size: 26 x 26 x n, receptive field: 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n*2),
            nn.ReLU()
        ) #input size: 26 x 26 x n, output size: 24 x 24 x n*2, receptive field: 3 + (3-1) * 1 = 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n*4),
            nn.ReLU()
        ) #input size: 24 x 24 x n*2, output size: 22 x 22 x n*4, receptive field: 5 + (3-1) * 1 = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) #input size: 22 x 22 x n*4, output size: 11 x 11 x n*4, receptive field: 7 + (3-1) * 1 = 8

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(n),
            nn.ReLU()
        ) #input size: 11 x 11 x n*4, output size: 11 x 11 x n, receptive field: 8 + (1-1)*2 = 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n*2),
            nn.ReLU()
        ) #input size: 11 x 11 x n, output size: 9 x 9 x n*2, receptive field: 8 + (3-1) * 2 = 12

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n*4),
            nn.ReLU()
        ) #input size: 9 x 9 x n*2, output size: 7 x 7 x n*4, receptive field: 12 + (3-1) * 2 = 16

               
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7) # 7>> 9... 
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 7 x 7 x n*4, output size: 1 x 1 x n*4, receptive field: 16

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        ) #input size: 1 x 1 x n*4, output size: 1 x 1 x 10, receptive field: 16 + (1-1) * 2 =16
        


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        x = self.pool1(x)
        x = self.convblock4(x)
        
        x = self.convblock5(x)
        x = self.convblock6(x)
        
        x = self.gap(x)
        x = self.convblock7(x)

        x = x.squeeze()
        # view(-1, 10)
        return F.log_softmax(x, dim=-1)
