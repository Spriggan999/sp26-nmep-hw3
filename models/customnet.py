"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a residual block for our ResNet18 architecture.

        Here is the expected network structure:
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=stride
        - batchnorm layer (Batchnorm2D)
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=1
        - batchnorm layer (Batchnorm2D)
        - shortcut layer:
            if either the stride is not 1 or the out_channels is not equal to in_channels:
                the shortcut layer is composed of two steps:
                - conv layer with
                    in_channels=in_channels, out_channels=out_channels, 1x1 kernel, stride=stride
                - batchnorm layer (Batchnorm2D)
            else:
                the shortcut layer should be an no-op

        All conv layers will have a padding of 1 and no bias term. To facilitate this, consider using
        the provided conv() helper function.
        When performing a forward pass, the ReLU activation should be applied after the first batchnorm layer
        and after the second batchnorm gets added to the shortcut.
        """
        ## YOUR CODE HERE

        ## Initialize the block with a call to super and make your conv and batchnorm layers.
        super(CustomBlock, self).__init__()
        # TODO: Initialize conv and batch norm layers with the correct parameters
        self.conv_1= nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv_2 = nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()


        ## Use some conditional logic when defining your shortcut layer
        ## For a no-op layer, consider creating an empty nn.Sequential()
        # TODO: Code here to initialize the shortcut layer
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

        ## END YOUR CODE

    def forward(self, x):
        """
        Compute a forward pass of this batch of data on this residual block.

        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        """
        ## YOUR CODE HERE
        ## TODO: Call the first convolution, batchnorm, and activation
        out_1 = self.conv_1(x)
        out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        ## TODO: Call the second convolution and batchnorm
        out_2 = self.conv_2(out_1)
        out_2 = self.bn2(out_2)

        ## TODO: Also call the shortcut layer on the original input
        out_3 = self.shortcut(x)

        ## TODO: Sum the result of the shortcut and the result of the second batchnorm
        ## and apply your activation
        out_4 = self.relu(out_2 + out_3)

        return out_4
        ## END YOUR CODE
        #pass  # Remove this line when you implement this function


class CustomNet(nn.Module):
    def __init__(self, num_classes=200):
        # Read the following, and uncomment it when you understand it, no need to add more code
        num_classes = num_classes
        super(CustomNet, self).__init__()
        self.in_channels = 64
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer12 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=1)
        self.layer22 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=1)
        self.layer32 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=1)
        self.layer42 = self.make_block(out_channels=512, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #pass  # Remove this line when you uncomment the above code

    def make_block(self, out_channels, stride):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        for stride in [stride, 1]:
            layers.append(CustomBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        #pass  # Remove this line when you uncomment the above code

    def forward(self, x):
        # Read the following, and uncomment it when you understand it, no need to add more code
        #x = F.relu(self.bn1(self.conv1(x)))
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer12(x)
        x = self.layer2(x)
        x = self.layer22(x)
        x = self.layer3(x)
        x = self.layer32(x)
        x = self.layer4(x)
        x = self.layer42(x)
        #x = F.avg_pool2d(x, 4)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        #x = F.Dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        return x
        #pass  # Remove this line when you uncomment the above code
