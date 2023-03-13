from .bert_model import Bert_base_lstm_Net, Bert_base_lstm_Net1
from .dna2vec_model import DNA2vec_CNN_Net, DNA2vec_CNN_LSTM_Net, DNA2vec_CNN2d_Net
from .electra_model import Electra_small_Net, Electra_DNA2vec_Net, Electra_lstm_Net, Electra_base_Net, Electra_large_Net
from .longformer_model import Longformer_base_Net, Longformer_base_lstm_Net, Longformer_base_2cnn_lstm_Net, \
    Longformer_base_1cnn_lstm_Net
from .longformer_model import Longformer_base_lstmBNdr_Net, Longformer_base_lstmInit_Net, \
    Longformer_base_lstm_BNpos_Net, Longformer_base_0cnn_lstm_Net
from .model import CNN2d_Net, CNN_Net
