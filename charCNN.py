#!/usr/bin/env python3

import torch
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, config):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(config['cnn_input_size'], 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    vec = torch.randn((5,10, 70, 20))
    model = CharCNN({'cnn_input_len':70})
    res = model(vec)