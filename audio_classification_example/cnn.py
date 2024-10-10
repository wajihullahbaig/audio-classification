# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 04:17:43 2024

@author: Acer
"""

import torch
from torch import nn



class CNNNetwork(nn.Module) :
    
    def __init__(self):
        super().__init__()
        
        # 4 convolutional blocks / flatten / linear / softmax
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 1, 
                    out_channels = 16, 
                    kernel_size = 3,
                    stride = 1,
                    padding = 2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 16, 
                    out_channels = 32, 
                    kernel_size = 3,
                    stride = 1,
                    padding = 2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 32, 
                    out_channels = 64, 
                    kernel_size = 3,
                    stride = 1,
                    padding = 2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv4 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 64, 
                    out_channels = 128, 
                    kernel_size = 3,
                    stride = 1,
                    padding = 2
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features = 128 * 5 * 4, out_features = 10)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(self,input_data):
        
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        prediction = self.softmax(logits)
        
        return prediction
 
    def _init_weights(self, m):
        # Lets initialize the CNN 
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)   

if __name__ == "__main__":

    model = CNNNetwork()    
    print(model)  