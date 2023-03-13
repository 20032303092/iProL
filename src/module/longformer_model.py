import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])


class Longformer_base_Net(nn.Module):
    """
    epoch 230 [167/167] total_loss = 0.30969953813595685 total=5323, TP=2278, TN=2350, FP=312, FN=383; precision=0.880, recall=0.856, Sn=0.856, Sp=0.883
    roc_auc: (test=0.941) average_precision: (test=0.944) accuracy: (test=0.869) matthews_corrcoef: (test=0.739) f1: (test=0.868)
    epoch 230 [42/42] total_loss = 0.40675214358738493 total=1331, TP=568, TN=544, FP=122, FN=97; precision=0.823, recall=0.854, Sn=0.854, Sp=0.817
    roc_auc: (test=0.902) average_precision: (test=0.909) accuracy: (test=0.835) matthews_corrcoef: (test=0.671) f1: (test=0.838)
    """

    def __init__(self, model_name, device, add_special_tokens=False):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(32)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.5)

        # in_features=384, out_features=128
        self.dense1 = nn.Linear(in_features=96, out_features=32)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

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

        x = self.cnn1(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

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


class Longformer_base_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(32)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=96, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

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

        x = self.cnn1(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn2(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn2(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp2(x)
        # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp2(x)
        # print("cnn_dp(x)", x.shape)

        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)

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


class Longformer_base_lstm_BNpos_Net(nn.Module):
    """
    修改BN到pooling之后dropout之前
    """

    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(32)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=96, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

        x = x.permute(0, 2, 1)

        x = self.cnn(x)
        # print("cnn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn1(x)
        # print("cnn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn2(x)
        # print("cnn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp2(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_bn2(x)
        # print("cnn_bn(x)", x.shape)
        # x = self.cnn_dp2(x)
        # print("cnn_dp(x)", x.shape)

        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)

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


class Longformer_base_lstmBNdr_Net(nn.Module):
    """ 和base相比， 加了 lstm_bn lstm_dp"""

    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(32)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        """ 和base相比， 加了 lstm_bn lstm_dp"""
        self.lstm_bn = nn.BatchNorm1d(3)
        self.lstm_dp = nn.Dropout(0.5)

        self.dense1 = nn.Linear(in_features=96, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

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

        x = self.cnn1(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn2(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn2(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp2(x)
        # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp2(x)
        # print("cnn_dp(x)", x.shape)

        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)
        x = self.lstm_bn(x)
        # print("lstm_bn", x.shape)
        x = self.lstm_dp(x)

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


class Longformer_base_lstmInit_Net(nn.Module):
    """
    h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
    c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)

    原文链接：https://blog.csdn.net/Cyril_KI/article/details/122557880
    """

    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn2 = nn.BatchNorm1d(32)
        self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=96, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

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

        x = self.cnn1(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn2(x)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn2(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp2(x)
        # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp2(x)
        # print("cnn_dp(x)", x.shape)

        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, x.shape[-1]).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, x.shape[-1]).to(self.device)
        x, (_, _) = self.lstm(x, (h_0, c_0))
        # print("lstm", x.shape)
        # x = F.relu(x)

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


class Longformer_base_2cnn_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.8)

        # self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        # self.cnn_bn2 = nn.BatchNorm1d(32)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=192, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.longformer", x.shape)  # (B,S,H) [4,81,768]

        x = x.permute(0, 2, 1)

        x = self.cnn(x)  # (N,C,L)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)

        x = self.cnn1(x)
        # print("cnn1(x)", x.shape)
        x = self.cnn_bn1(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp1(x)
        # print("cnn_mp1(x)", x.shape)
        x = self.cnn_dp1(x)
        # print("cnn_dp(x)", x.shape)

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)
        #
        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)

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


class Longformer_base_1cnn_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.8)

        # self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        # self.cnn_bn1 = nn.BatchNorm1d(64)
        # self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp1 = nn.Dropout(0.8)

        # self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        # self.cnn_bn2 = nn.BatchNorm1d(32)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=128, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=1120, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.longformer", x.shape)  # (B,S,H) [4,81,768]

        x = x.permute(0, 2, 1)

        x = self.cnn(x)  # (N,C,L)
        # print("cnn(x)", x.shape)
        x = self.cnn_bn(x)
        # print("cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.cnn_mp(x)
        # print("cnn_mp(x)", x.shape)
        x = self.cnn_dp(x)
        # print("cnn_dp(x)", x.shape)

        # x = self.cnn1(x)
        # # print("cnn1(x)", x.shape)
        # x = self.cnn_bn1(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp1(x)
        # # print("cnn_mp1(x)", x.shape)
        # x = self.cnn_dp1(x)
        # # print("cnn_dp(x)", x.shape)

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)
        #
        x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)

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

class Longformer_base_0cnn_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = LongformerConfig.from_pretrained(self.model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        self.longformer = LongformerModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        # self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=8, stride=2)
        # self.cnn_bn = nn.BatchNorm1d(128)
        # self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp = nn.Dropout(0.8)

        # self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2)
        # self.cnn_bn1 = nn.BatchNorm1d(64)
        # self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp1 = nn.Dropout(0.8)

        # self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        # self.cnn_bn2 = nn.BatchNorm1d(32)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=768, hidden_size=16, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        self.dense1 = nn.Linear(in_features=1296, out_features=64)  # first 512
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=self.add_special_tokens,
                                        padding=True)
        encoded_inputs.to(self.device)
        x = self.longformer(**encoded_inputs)[0]
        # print("self.longformer", x.shape)  # (B,S,H) [4,81,768]

        # x = x.permute(0, 2, 1)

        # x = self.cnn(x)  # (N,C,L)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp(x)
        # # print("cnn_mp(x)", x.shape)
        # x = self.cnn_dp(x)
        # # print("cnn_dp(x)", x.shape)

        # x = self.cnn1(x)
        # # print("cnn1(x)", x.shape)
        # x = self.cnn_bn1(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp1(x)
        # # print("cnn_mp1(x)", x.shape)
        # x = self.cnn_dp1(x)
        # # print("cnn_dp(x)", x.shape)

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)
        #
        # x = x.permute(0, 2, 1)  # B,S,H
        # print("x.permute", x.shape)

        x, (_, _) = self.lstm(x)
        # print("lstm", x.shape)
        # x = F.relu(x)

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

if __name__ == '__main__':
    import src.config_iProL as config

    """
    longformer_base_lstm demo
    """
    input = [
        'A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A C G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A G T C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A C G A A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
    ]
    print(len(input[0]))  # 81
    model = Longformer_base_0cnn_lstm_Net(config.model_name, 'cpu',
                                config.add_special_tokens)
    # print(model)
    model.to("cpu")
    # out = torch.randint(0, 2, (1, 4))
    out = model(input)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    y = torch.tensor([0, 0, 1, 1]).float()
    print(out)
    print(out.shape)
