import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig, BertTokenizer  # , DNATokenizer

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])


class Bert_base_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = BertConfig.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.longformer = BertModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

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


class Bert_base_lstm_Net1(nn.Module):
    """
    由cnn、bn、relu、maxpooling、dr的顺序，
    变为-->
    cnn、relu、maxpooling、bn、dr
    bn在池化后
    """

    def __init__(self, model_name, device, add_special_tokens=False, bidirectional=False, num_layers=1):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = BertConfig.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.longformer = BertModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

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


if __name__ == '__main__':
    pass
