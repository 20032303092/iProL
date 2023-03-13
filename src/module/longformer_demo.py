import src.config_iProL as config
from src.module import Longformer_base_lstm_Net
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

model_name = config.model_name

model_config = LongformerConfig.from_pretrained(model_name)
tokenizer = LongformerTokenizer.from_pretrained(model_name)
longformer = LongformerModel.from_pretrained(model_name, config=model_config)  # (B,S,256)

input = [
    'ATG AAA AAC AAG AAT ACA ACC ACG ACT AGA AGC AGG AGT ATA ATC ATG ATT CAC GAT CAG TAC ATG CAC GAT CAG TAC CAG TAC',
    'CTG CAA CAC CAG CAT CCA CCC CCG CCT CGA CGC CGG CGT CTA CTC CTG CTT CAC GAT CAG TAC ATG CAC GAT CAG TAC CAG TAC',
    'GTG GAA GAC GAG GAT GCA GCC GCG GCT GGA GGC GGG GGT GTA GTC GTG GTT GAC GAT CAG TAC ATG CAC GAT CAG TAC CAG TAC',
    'TTG TAA TAC TAG TAT TCA TCC TCG TCT TGA TGC TGG TGT TTA TTC TTG TTT TAC GAT CAG TAC ATG CAC GAT CAG TAC CAG TAC',
]
encoded_inputs = tokenizer(input, return_tensors='pt', add_special_tokens=True, padding=True)
print(encoded_inputs)

for item in encoded_inputs['input_ids']:
    decoder = tokenizer.decode(item)
    print(decoder)

x = longformer(**encoded_inputs)

tensor = x[0]
