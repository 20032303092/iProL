import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

_current_path = os.path.dirname(__file__)
root_path = os.path.abspath(_current_path).split('src')
sys.path.extend([root_path[0] + 'src'])


class CNN_Net(nn.Module):
    def __init__(self):
        # super(CNNNet, self).__init__()
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=5, out_channels=128, kernel_size=4, stride=1)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=30, stride=2)
        self.cnn_dp = nn.Dropout(0.5)
        self.dense1 = nn.Linear(in_features=3200, out_features=128)
        self.dense1_dp = nn.Dropout(0.5)
        self.dense2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # print("input", x.shape)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)

        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)
        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        # print("dense1(x)", x.shape)
        x = F.relu(x)
        x = self.dense1_dp(x)
        x = F.relu(x)
        x = self.dense2(x)
        # print(x)
        # x = F.softmax(x, dim=-1)
        # print("dense2(x)", x.shape)
        x = torch.sigmoid(x)
        # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x


class CNN2d_Net(nn.Module):
    def __init__(self):
        # super(CNNNet, self).__init__()
        super().__init__()
        # self.emd = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, _weight=None)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 40), stride=(1, 2))
        self.cnn_bn = nn.BatchNorm2d(3)
        self.cnn_mp = nn.MaxPool2d(kernel_size=(10, 15), stride=3)
        self.cnn_dp = nn.Dropout(0.2)
        self.dense1 = nn.Linear(in_features=2208, out_features=64)
        self.dense1_dp = nn.Dropout(0.2)
        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        print("input", x.shape)
        x = x.unsqueeze(1)
        print("input", x.shape)
        x = self.cnn(x)
        # print("cnn(x)", x.shape)
        # x = self.cnn_bn(x)
        print("cnn_bn(x)", x.shape)
        x = F.relu(x)

        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)
        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        # print("dense1(x)", x.shape)
        x = F.relu(x)
        x = self.dense1_dp(x)
        x = F.relu(x)
        x = self.dense2(x)
        # print(x)
        # x = F.softmax(x, dim=-1)
        # print("dense2(x)", x.shape)
        x = torch.sigmoid(x)
        # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x
