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


class DNA2vec_CNN_Net(nn.Module):
    def __init__(self, kmer, ebd_file):
        super().__init__()
        embedding_matrix = torch.as_tensor(np.load(ebd_file))
        ebd_kmer = int(math.log(embedding_matrix.shape[0], 4))  # kmer
        if kmer != ebd_kmer:
            raise ValueError(f"ebd_kmer should be {ebd_kmer}, please check it in config file!!!")
        num_embeddings = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        print(f"kmer: {ebd_kmer}, embedding_dim: {embedding_dim}")
        self.emd = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                _weight=embedding_matrix)
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(self.cnn.out_channels)
        self.cnn_mp = nn.MaxPool1d(kernel_size=4, stride=1)
        self.cnn_dp = nn.Dropout(0.2)

        # # kernel_size=3, stride=2
        # self.cnn1 = nn.Conv1d(in_channels=self.cnn.out_channels, out_channels=32, kernel_size=8, stride=2)
        # self.cnn_bn1 = nn.BatchNorm1d(self.cnn1.out_channels)
        # self.cnn_mp1 = nn.MaxPool1d(kernel_size=4, stride=1)
        # self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=self.cnn.out_channels, out_channels=16, kernel_size=8, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(self.cnn2.out_channels)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.2)

        # in_features=384, out_features=128
        self.dense1 = nn.Linear(in_features=176, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.2)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # print(x[0])
        # print("input", x.shape)
        # print(x.dtype)
        # x = self.emd(x.long())
        # print(x.dtype)
        x = self.emd(x.long())
        # print("emd(x)", x.shape)
        # print(x.dtype)
       
        x = x.permute(0, 2, 1)

        x = self.cnn(x.float())
        # print("cnn(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)

        # x = self.cnn1(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn1(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp1(x)
        # # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp1(x)
        # # print("cnn_dp(x)", x.shape)

        x = self.cnn2(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn2(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp2(x)
        # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp2(x)
        # print("cnn_dp(x)", x.shape)

        x = x.reshape([x.shape[0], -1])

        x = self.dense1(x)
        # print("dense1(x)", x.shape)
        x = F.relu(x)
        # x = self.dense1_dp(x)

        x = self.dense2(x)
        # print(x)
        # x = F.softmax(x, dim=-1)
        # print("dense2(x)", x.shape)
        x = torch.sigmoid(x)
        # # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x


class DNA2vec_CNN2d_Net(nn.Module):
    def __init__(self, kmer, ebd_file):
        # super(CNNNet, self).__init__()
        super().__init__()
        # embedding_matrix = torch.as_tensor(np.load(f"../embedding/embedding_matrix_{species}_{flag}_kmer{kmer}.npy"))
        ebd_path = os.path.join(_current_path, f"../embedding/embedding_matrix_{ebd_file}.npy")
        embedding_matrix = torch.as_tensor(np.load(ebd_path))
        embedding_dim = embedding_matrix.shape[1]
        print("embedding_dim", embedding_dim)
        layer = {
            32: [[3, 4, 1], [(30, 20), (2, 1)]],
            50: [[8, 3, 1], [15, 2]],
            100: [[8, 3, 1], [15, 2]]
        }
        self.emd = nn.Embedding(num_embeddings=int(math.pow(4, kmer)), embedding_dim=embedding_dim,
                                _weight=embedding_matrix)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=layer[embedding_dim][0][0],
                             kernel_size=layer[embedding_dim][0][1],
                             stride=layer[embedding_dim][0][2])
        self.cnn_bn = nn.BatchNorm2d(layer[embedding_dim][0][0])
        self.cnn_mp = nn.MaxPool2d(kernel_size=layer[embedding_dim][1][0], stride=layer[embedding_dim][1][1])
        self.cnn_dp = nn.Dropout(0.5)

        in_features = {
            32: {3: 720, 4: 1984, 5: 1984, 6: 1920},
            50: {3: 4352, 4: 4216, 5: 4216, 6: 4080},
            100: {3: 10752, 4: 10416, 5: 10416, 6: 10080}
        }
        self.dense1 = nn.Linear(in_features=in_features[embedding_dim][kmer], out_features=256)
        self.dense1_dp = nn.Dropout(0.5)
        self.dense3 = nn.Linear(in_features=256, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # print("input", x.shape)
        x = self.emd(x.long())
        # print("emd(x)", x.shape)
        # print(x.dtype)

        x = x.unsqueeze(1)
        x = self.cnn(x.float())
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
        x = self.dense3(x)
        x = self.dense2(x)
        # print(x)
        # x = F.softmax(x, dim=-1)
        # print("dense2(x)", x.shape)
        x = torch.sigmoid(x)
        # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x


class DNA2vec_CNN_LSTM_Net(nn.Module):
    def __init__(self, kmer, ebd_file, bidirectional=False):
        # super(CNNNet, self).__init__()
        super().__init__()
        # embedding_matrix = torch.as_tensor(np.load(f"../embedding/embedding_matrix_{species}_{flag}_kmer{kmer}.npy"))
        ebd_path = os.path.join(_current_path, f"../embedding/embedding_matrix_{ebd_file}.npy")
        embedding_matrix = torch.as_tensor(np.load(ebd_path))
        embedding_dim = embedding_matrix.shape[1]
        print("embedding_dim", embedding_dim)
        self.emd = nn.Embedding(num_embeddings=int(math.pow(4, kmer)), embedding_dim=embedding_dim,
                                _weight=embedding_matrix)
        out_channel = 16
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channel, kernel_size=2, stride=1)
        self.cnn_bn = nn.BatchNorm1d(out_channel)
        self.cnn_mp = nn.MaxPool1d(kernel_size=2, stride=1)
        self.cnn_dp = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=out_channel, hidden_size=4, num_layers=2,
                            batch_first=True, bidirectional=self.bidirectional)

        in_features = {3: 308, 4: 304, 5: 300, 6: 296}
        if self.bidirectional:
            in_features.update({kmer: in_features[kmer] * 2})
        self.dense1 = nn.Linear(in_features=in_features[kmer], out_features=128)
        self.dense1_dp = nn.Dropout(0.5)
        self.dense2 = nn.Linear(in_features=128, out_features=32)
        self.dense2_dp = nn.Dropout(0.5)
        self.dense3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        # print("input", x.shape)
        x = self.emd(x.long())
        # print("emd(x)", x.shape)  B,S,H
        # print(x.dtype)

        x = x.permute(0, 2, 1)  # B,H,S
        x = self.cnn(x.float())
        # print("cnn(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)

        # x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)

        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        # print("dense1(x)", x.shape)
        x = F.relu(x)
        x = self.dense1_dp(x)

        x = self.dense2(x)
        # print(x)
        x = F.relu(x)
        x = self.dense2_dp(x)
        # x = F.softmax(x, dim=-1)
        # print("dense2(x)", x.shape)

        x = self.dense3(x)
        x = torch.sigmoid(x)
        # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x


if __name__ == '__main__':
    torch.random.manual_seed(1)
    """
    CNNNet demo
    """
    # X = torch.randn(4, 81, 4)
    # out = CNN_Net()(X)
    # # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # y = torch.tensor([0, 0, 1, 1]).float()
    # print(out)
    # print(out.shape)
    # loss = criterion(out, y)
    # print(loss.item())
    """
    CNN2d_Net demo
    """
    # X = torch.randn(4, 81, 64)
    # out = CNN2d_Net()(X)
    # # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # y = torch.tensor([0, 0, 1, 1]).float()
    # print(out)
    # print(out.shape)
    # loss = criterion(out, y)
    # print(loss.item())
    """
    DNA2vec_CNN_Net demo
    """
    # X = torch.randint(0, 64, (4, 81))
    # print(X.shape)
    # out = DNA2vec_CNN_Net(3)(X)
    # # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # y = torch.tensor([0, 0, 1, 1]).float()
    # print(out)
    # loss = criterion(out, y)
    # print(loss.item())
    """
    CNN2d_Net demo
    """
    # kmer = 3
    # flag = f"E_coil_32d10c_kmer{kmer}"
    # X = torch.randint(0, 4 ** kmer, size=(4, 81 - kmer + 1))
    # out = DNA2vec_CNN2d_Net(kmer=kmer, ebd_file=flag)(X)
    # # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # y = torch.tensor([0, 0, 1, 1]).float()
    # print(out)
    # print(out.shape)
    # loss = criterion(out, y)
    # print(loss.item())
    """
    DNA2vec_CNN_LSTM_Net demo
    """
    kmer = 5
    print("kmer", kmer)
    flag = f"E_coil_32d10c_kmer{kmer}"
    X = torch.randint(0, 64, (4, 81))
    print(X.shape)
    out = DNA2vec_CNN_LSTM_Net(kmer, flag)(X)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    y = torch.tensor([0, 0, 1, 1]).float()
    print(out)
    loss = criterion(out, y)
    print(loss.item())
