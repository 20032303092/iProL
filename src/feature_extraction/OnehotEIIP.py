import numpy as np
from joblib import Parallel, delayed

"""
在ComposeFeature类中有一样的实现
"""


class OnehotEIIP:
    one_hot_eiip_dict = {'a': [0, 0, 0, 1, 0.1260],
                         'c': [0, 0, 1, 0, 0.1340],
                         'g': [0, 1, 0, 0, 0.0806],
                         't': [1, 0, 0, 0, 0.1335]}

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in seq:
            seq_one.append(self.one_hot_eiip_dict[str(s).lower()])
        one_hot_vec = np.array(seq_one)
        return one_hot_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("onehot_eiip Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out, np.float32)

    def get_params(self):
        print("Onehot_EIIP Params", self.__dict__)


if __name__ == '__main__':
    print(OnehotEIIP().get_fea("ATGCA"))
