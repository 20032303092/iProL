import numpy as np
from transformers import ElectraTokenizer, ElectraConfig, ElectraForPreTraining


# device = 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# kmer = Kmer(k=[1], stride=1, return_type='seq')
# identify_train_kmer_seq = kmer.run_Kmer(dataset_identify_train["Pro_seq"].tolist())
# identify_test_kmer_seq = kmer.run_Kmer(dataset_identify_test["Pro_seq"].tolist())
#
# model_name = './pre-model/' + 'electra/' + 'electra-small-discriminator'
#
# config = ElectraConfig.from_pretrained(model_name)
# tokenizer = ElectraTokenizer.from_pretrained(model_name)
# model = ElectraForPreTraining.from_pretrained(model_name, config=config).to(device)
#
# identify_train_kmer_seq_encoded_inputs = tokenizer(identify_train_kmer_seq, return_tensors='pt',
#                                                    add_special_tokens=False, ).to(device)
# identify_test_kmer_seq_encoded_inputs = tokenizer(identify_test_kmer_seq, return_tensors='pt', add_special_tokens=False)
# identify_train_kmer_seq_vec = model(**identify_train_kmer_seq_encoded_inputs).to(device)
# identify_test_kmer_seq_vec = model(**identify_test_kmer_seq_encoded_inputs).to(device)
# print("X_tensor:", identify_train_kmer_seq_vec[0].shape)  # (Batch_size,2651,768) (B,S)
# print("X_tensor:", identify_test_kmer_seq_vec[0].shape)  # (Batch_size,2651,768) (B,S)


class ElectraEmbedding:
    def __init__(self, model_name, batch_size, device, **model_kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model_kwargs = model_kwargs
        self.config = ElectraConfig.from_pretrained(model_name)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForPreTraining.from_pretrained(model_name, config=self.config).to(device)

    def run_fea(self, kmer_seq_list):
        kmer_seq_vec = []
        return_tensors = 'pt'
        add_special_tokens = False
        if self.model_kwargs.get('return_tensors') is not None:
            return_tensors = self.model_kwargs.get('return_tensors')
        if self.model_kwargs.get('add_special_tokens') is not None:
            add_special_tokens = self.model_kwargs.get('add_special_tokens')

        for str_idx in range(0, len(kmer_seq_list), self.batch_size):
            batch = kmer_seq_list[str_idx:str_idx + self.batch_size]
            kmer_seq_encoded_inputs_batch = self.tokenizer(batch, return_tensors=return_tensors,
                                                           add_special_tokens=add_special_tokens, ).to(self.device)
            kmer_seq_vec_batch = self.model(**kmer_seq_encoded_inputs_batch)[0].to(self.device)
            kmer_seq_vec_batch = kmer_seq_vec_batch.to(self.device).tolist()
            kmer_seq_vec.extend(kmer_seq_vec_batch)

        kmer_seq_vec = np.array(kmer_seq_vec)
        # check
        print("check[ElectraEmbedding]:", len(kmer_seq_list), len(kmer_seq_vec), kmer_seq_vec.shape)

        return kmer_seq_vec

    def get_params(self):
        print("ElectraEmbedding Params", self.__dict__)


if __name__ == '__main__':
    from Kmer import Kmer
    import torch

    # usage:
    model_name = '../pre-model/' + 'electra/' + 'electra-small-discriminator'
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')

    x = ["ATGCATGACTGTACGTAAACGTAA",
         "ATGCATGACTGTGATTAGACGTAA",
         "ATTCATGACTGTTGAAGAACGTAA",
         "ATTCATGACTGTTGAAGAACGTAA"]

    kmer = Kmer(k=2, stride=1, return_type="seq")
    kmer_seq_list = kmer.run_fea(x)
    print(kmer_seq_list)
    embedding = ElectraEmbedding(model_name, device=device, batch_size=4, add_special_tokens=True)
    kmer_seq_vec = embedding.run_fea(kmer_seq_list)
    print(kmer_seq_vec)
