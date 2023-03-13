import math
import os
import sys

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
from transformers import ElectraModel, ElectraTokenizer, ElectraConfig

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])


class Electra_small_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        # model_name = 'pre-model/' + 'electra-base-discriminator'
        self.model_name = model_name

        self.config = ElectraConfig.from_pretrained(self.model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.electra = ElectraModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.5)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(32)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.5)

        # self.cnn2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
        # self.cnn_bn2 = nn.BatchNorm1d(16)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.dense1 = nn.Linear(in_features=512, out_features=256)
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.electra(**encoded_inputs)[0]
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

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)

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


class Electra_base_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = ElectraConfig.from_pretrained(self.model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.electra = ElectraModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.5)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(32)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.5)

        # self.cnn2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
        # self.cnn_bn2 = nn.BatchNorm1d(16)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.dense1 = nn.Linear(in_features=512, out_features=256)
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.electra(**encoded_inputs)[0]
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

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)

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


class Electra_large_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = ElectraConfig.from_pretrained(self.model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.electra = ElectraModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.5)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(32)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.5)

        # self.cnn2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
        # self.cnn_bn2 = nn.BatchNorm1d(16)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.dense1 = nn.Linear(in_features=512, out_features=256)
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.electra(**encoded_inputs)[0]
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

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)

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


class Electra_lstm_Net(nn.Module):
    def __init__(self, model_name, device, add_special_tokens=False):
        super().__init__()
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = ElectraConfig.from_pretrained(self.model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.electra = ElectraModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.cnn = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.cnn_bn = nn.BatchNorm1d(128)
        self.cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp = nn.Dropout(0.5)

        self.cnn1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.cnn_bn1 = nn.BatchNorm1d(32)
        self.cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.cnn_dp1 = nn.Dropout(0.5)

        # self.cnn2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
        # self.cnn_bn2 = nn.BatchNorm1d(16)
        # self.cnn_mp2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.cnn_dp2 = nn.Dropout(0.5)

        self.dense1 = nn.Linear(in_features=512, out_features=256)
        self.dense1_dp = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.electra(**encoded_inputs)[0]
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

        # x = self.cnn2(x)
        # # print("cnn(x)", x.shape)
        # x = self.cnn_bn2(x)
        # # print("cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.cnn_mp2(x)
        # # print("cnn_mp(x)", x.shape)
        # # x = self.cnn_dp2(x)
        # # print("cnn_dp(x)", x.shape)

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


class Electra_DNA2vec_Net(nn.Module):
    def __init__(self, model_name, ebd_file, device, add_special_tokens=False, kmer=3):
        super().__init__()
        self.device = device
        """
        ELECTRA PART
        """
        self.add_special_tokens = add_special_tokens
        self.model_name = model_name
        # model_name = 'pre-model/' + 'longformer-base-4096'
        self.config = ElectraConfig.from_pretrained(self.model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.electra = ElectraModel.from_pretrained(self.model_name, config=self.config)  # (B,S,256)

        self.ele_cnn = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.ele_cnn_bn = nn.BatchNorm1d(128)
        self.ele_cnn_mp = nn.MaxPool1d(kernel_size=3, stride=1)
        self.ele_cnn_dp = nn.Dropout(0.5)

        self.ele_cnn1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.ele_cnn_bn1 = nn.BatchNorm1d(32)
        self.ele_cnn_mp1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.ele_cnn_dp1 = nn.Dropout(0.5)

        self.ele_dense1 = nn.Linear(in_features=512, out_features=256)
        self.ele_dense1_dp = nn.Dropout(0.5)

        self.ele_dense2 = nn.Linear(in_features=256, out_features=1)

        """
        DNA2VEC PART
        """
        embedding_matrix = torch.as_tensor(np.load(f"../embedding/embedding_matrix_{ebd_file}.npy"))
        embedding_dim = int((ebd_file[7:]).split('d')[0])
        # embedding_dim = 100
        print("embedding_dim", embedding_dim)
        self.emd = nn.Embedding(num_embeddings=int(math.pow(4, kmer)), embedding_dim=embedding_dim,
                                _weight=embedding_matrix)
        self.dna_cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3, stride=1)
        self.dna_cnn_bn = nn.BatchNorm1d(32)
        self.dna_cnn_mp = nn.MaxPool1d(kernel_size=8, stride=2)
        self.dna_cnn_dp = nn.Dropout(0.8)

        in_features = {3: 1120, 4: 1088, 5: 1088, 6: 1056}
        self.dna_dense1 = nn.Linear(in_features=in_features[kmer], out_features=128)
        # self.dna_dense1 = nn.Linear(in_features=1088, out_features=32)
        self.dna_dense1_dp = nn.Dropout(0.7)
        self.dna_dense2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        encoded_inputs = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)
        encoded_inputs.to(self.device)
        # print(encoded_inputs)
        # for item in encoded_inputs['input_ids']:
        #     decoder = self.tokenizer.decode(item)
        #     print(decoder)
        x = self.electra(**encoded_inputs)[0]
        # print("self.electra", x.shape)  # (B,S,H)

        x = x.permute(0, 2, 1)

        x = self.ele_cnn(x)
        # print("ele_cnn(x)", x.shape)
        x = self.ele_cnn_bn(x)
        # print("ele_cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.ele_cnn_mp(x)
        # print("ele_cnn_mp(x)", x.shape)
        x = self.ele_cnn_dp(x)
        # print("ele_cnn_dp(x)", x.shape)

        x = self.ele_cnn1(x)
        # print("ele_cnn(x)", x.shape)
        x = self.ele_cnn_bn1(x)
        # print("ele_cnn_bn(x)", x.shape)
        x = F.relu(x)
        x = self.ele_cnn_mp1(x)
        # print("ele_cnn_mp(x)", x.shape)
        x = self.ele_cnn_dp1(x)
        # print("ele_cnn_dp(x)", x.shape)

        # x = self.ele_cnn2(x)
        # # print("ele_cnn(x)", x.shape)
        # x = self.ele_cnn_bn2(x)
        # # print("ele_cnn_bn(x)", x.shape)
        # x = F.relu(x)
        # x = self.ele_cnn_mp2(x)
        # # print("ele_cnn_mp(x)", x.shape)
        # # x = self.ele_cnn_dp2(x)
        # # print("ele_cnn_dp(x)", x.shape)

        x = x.reshape([x.shape[0], -1])

        x = self.ele_dense1(x)
        # print("ele_dense1(x)", x.shape)
        x = F.relu(x)
        # x = self.ele_dense1_dp(x)

        x = self.ele_dense2(x)
        # print(x)
        # x = F.softmax(x, dim=-1)
        # print("ele_dense2(x)", x.shape)
        x = torch.sigmoid(x)
        # # print(x)
        x = x.view(-1)
        # print("out", x.shape)
        return x


if __name__ == '__main__':
    import src.config as config

    input = [
        'A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A C G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A G T C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
        'A C G A A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C A T G C A C G A T C A G T A C C A G T A C',
    ]

    out = torch.randint(0, 2, (1, 4))
    # model_name = r"../pre-model/electra/electra-small-discriminator"

    model = Electra_small_Net(config.model_name, 'cuda', True)
    model.to('cuda')
    y_hat = model(input)
    print(y_hat)
