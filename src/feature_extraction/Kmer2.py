import numpy as np
from joblib import Parallel, delayed


class Kmer2:
    kme2_dict = {
        'aa': [0, 0, 0, 0], 'ca': [0, 1, 0, 0], 'ga': [1, 0, 0, 0], 'ta': [1, 1, 0, 0],
        'ac': [0, 0, 0, 1], 'cc': [0, 1, 0, 1], 'gc': [1, 0, 0, 1], 'tc': [1, 1, 0, 1],
        'ag': [0, 0, 1, 0], 'cg': [0, 1, 1, 0], 'gg': [1, 0, 1, 0], 'tg': [1, 1, 1, 0],
        'at': [0, 0, 1, 1], 'ct': [0, 1, 1, 1], 'gt': [1, 0, 1, 1], 'tt': [1, 1, 1, 1],
    }

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in range(0, len(seq) - 1):
            seq_one.append(self.kme2_dict[str(seq[s:s + 2]).lower()])
        one_hot_vec = np.array(seq_one)
        return one_hot_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("one-hot Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("Kmer2 Params", self.__dict__)
