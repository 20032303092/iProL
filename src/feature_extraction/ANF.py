import time

import numpy as np
from joblib import Parallel, delayed


class ANF:
    def __init__(self, direction="left", n_jobs=1):
        self.direction = direction
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        # time.sleep(np.random.randint(2, 10))
        if self.direction == "left":
            step = 1
        else:
            step = -1
        vec = []
        start0 = 0 if step == 1 else len(seq) - 1
        end0 = len(seq) if step == 1 else -1
        for idx in range(start0, end0, step):
            s = seq[idx]
            count = 0
            l = idx + 1 if step == 1 else (len(seq) - idx)
            # print(l)
            start = 0 if step == 1 else len(seq) - 1
            end = (idx + 1) if step == 1 else idx - 1
            for i in range(start, end, step):
                if s == seq[i]:
                    count += 1
            vec.append(count / l)
        # print(seq, vec)
        return np.array(vec)

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("ANF Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("ANF Params:", self.__dict__)


if __name__ == '__main__':
    anf = ANF(n_jobs=3, direction="right")
    # print(anf.direction)
    # vec = anf.get_anf("AGCTACGTAAGGTT")
    # print(vec)
    l = ['AAACAAACAAAC',
         'TTCTTCTTCTTC',
         'GGGGGGGGGGGG',
         'AACCTTGGAACC',
         'AAACCCGGGTTT']
    vec_list = anf.run_fea(l)
    print(vec_list)
    print(vec_list)
    anf = ANF(n_jobs=3, direction="left")
    # print(anf.direction)
    # vec = anf.get_anf("AGCTACGTAAGGTT")
    # print(vec)
    l = ['AAACAAACAAAC',
         'TTCTTCTTCTTC',
         'GGGGGGGGGGGG',
         'AACCTTGGAACC',
         'AAACCCGGGTTT']
    vec_list = anf.run_fea(l)
    print(vec_list)
    print(vec_list)
