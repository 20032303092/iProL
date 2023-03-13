import numpy as np
from joblib import Parallel, delayed


class NCP:
    ncp_dict = {'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]}

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in seq:
            seq_one.append(self.ncp_dict[str(s).lower()])
        one_hot_vec = np.array(seq_one)
        return one_hot_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("NCP Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(o)

    def get_params(self):
        print("NCP Params", self.__dict__)


if __name__ == '__main__':
    ncp = NCP()
    print(ncp.get_fea("ATGC"))
