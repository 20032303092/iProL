import os.path
import time

import torch
print("『PBC note』Longformer_base_Net相比base，没有LSTM")

pred = "_Pro"
now_time = time.strftime('%m%d%H%M%S', time.localtime()) + pred
train_test_random = 0  # not work
_current_path = os.path.dirname(__file__)
_py_name = os.path.basename(__file__)

"""
dataset config
"""
version = ""
sheet_name = "dataset"
# split dataset
cv = 5
random_state = 7
# feature exaction for longformer
kmer = 2

"""
train config
"""
epochs = 250
batch_size = 32
lr = 0.0005
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
pre-model config
"""
pre_model = True
add_special_tokens = False
not_fine_tuning = True
_pre_model_root = "../pre-model/"
_pre_model_dir = "longformer/"
pre_model_file = "longformer-base-4096"
model_name = os.path.abspath(os.path.join(_current_path, _pre_model_root, _pre_model_dir, pre_model_file))
# print("config[pre_model_name]:", model_name)

"""
lstm
"""
lstm = False
bidirectional = True
num_layers = 1

"""
dna2vec config
"""
dna2vec = False
_ebd_tail = ""

"""
result config
"""
pc_name = _py_name.split('.')[0][-4:]
_ex_root = '../result/'
_result_dir = "longformer/"

# print("config[ex_result_dir]:", os.path.abspath(ex_result_dir))
_train_part = f"_train[kmer{kmer}_bs{batch_size}_lr{lr}_epoch{epochs}]"
_pre_model_part = ""
if pre_model:
    _model_level = f'base'
    _pre_model_part = f"_pm[ast{add_special_tokens}_nft{not_fine_tuning}_level{_model_level}]"
_dna2vec_part = ""
if dna2vec:
    _dna2vec_part = f"_d2v[{_ebd_tail}]"
_lstm_part = ""
if lstm:
    _lstm_part = f"_lstm[bi{bidirectional}_nl{num_layers}]"
_other_part = ""
file_tail = _train_part + _pre_model_part + _dna2vec_part + _lstm_part + _other_part

# loss pic
loss_pic_name = f"loss{pc_name}" + file_tail + ".png"
# score csv
train_score_name = f"train_score{pc_name}" + file_tail + ".csv"
test_score_name = f"test_score{pc_name}" + file_tail + ".csv"

# result dir
ex_result_dir = os.path.join(_current_path, _ex_root, _result_dir, file_tail, pc_name + '_' + now_time)

"""
encoding config
"""
# —————————————————————————————————————————————————————————————————————————————————————————————————
#   encoding   |             encoding_params              |              description
# kmer         |                                          | format:
# one-hot      |                                          | format:
# kmer_one_hot | kmer,stride                              | format:
# kmer_bert    | kmer,stride                              | format: ATGC ATCG ATGC
# —————————————————————————————————————————————————————————————————————————————————————————————————
X_encoding = 'kmer'
