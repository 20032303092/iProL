import numpy as np
from joblib import Parallel, delayed


class NCPEIIP:
    ncp_eiip_dict = {'a': [1, 1, 1, 0.1260], 'c': [0, 1, 0, 0.1340], 'g': [1, 0, 0, 0.0806], 't': [0, 0, 1, 0.1335]}

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in seq:
            seq_one.append(self.ncp_eiip_dict[str(s).lower()])
        one_hot_vec = np.array(seq_one)
        return one_hot_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("ncp_eiip Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("NCP_EIIP Params", self.__dict__)


if __name__ == '__main__':
    ne = NCPEIIP()
    seq_list = ["AGCTGCA", "AGCTGCA", "AGCTGCA", "AGCTGCA"]
    vec = ne.run_fea(seq_list)
    import torch

    print(vec)
    data = torch.tensor(vec).float()
    print(data)
    print(data.shape)
