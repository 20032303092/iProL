import numpy as np
from joblib import Parallel, delayed


class Onehot:
    one_hot_seq = {'a': [0, 0, 0, 1], 'c': [0, 0, 1, 0], 'g': [0, 1, 0, 0], 't': [1, 0, 0, 0]}

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in seq:
            seq_one.append(self.one_hot_seq[str(s).lower()])
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
        print("Onehot Params", self.__dict__)


if __name__ == '__main__':
    oh = Onehot()
    seq_list = ["AAAA", "CCCC", "GGGG", "TTTT"]
    vec = oh.run_fea(seq_list)
    print(vec)
