import numpy as np
from joblib import Parallel, delayed


class CKSNAP:
    """
    组成的核苷酸对("GC"，"CC"，"AT"，"AA"，"AG"，"AC"，"GA"，"CT"，"CA"，"GG"，"GT"，"CG"，"TG"，"TG"，"TT"，"TA")是由k个间隔的两个核苷酸组成的
    计算被任意k个核酸分隔的核苷酸bp的出现频率。即计算两个核苷酸分别位于`i`和`i + K + 1`位置的核苷酸对的频率。
    format:
    (F_GC/N,F_CC/N,...,F_TA/N)
    F_XX：frequency of 核苷酸对XX
    """

    def __init__(self, K=range(0, 6), n_jobs=1):
        self.n_jobs = n_jobs
        self.K = K
        self.ATGC_base = {"AA": 0, "AC": 0, "AG": 0, "AT": 0,
                          "CA": 0, "CC": 0, "CG": 0, "CT": 0,
                          "GA": 0, "GC": 0, "GG": 0, "GT": 0,
                          "TA": 0, "TC": 0, "TG": 0, "TT": 0}

    def get_CKSNAP_item(self, seq, k: int):
        ATGC = self.ATGC_base.copy()
        # print(ATGC)
        N = len(seq) - k - 1
        for idx in range(0, N):
            atgc = seq[idx] + seq[idx + k + 1]
            # print(atgc)
            if atgc.__contains__("N"):
                N = N - 1
            else:
                ATGC[atgc] += 1
        F_atgc = [v / N for v in ATGC.values()]
        # print(F_atgc)
        return F_atgc

    def get_fea(self, seq):
        CKSNAP = []
        for k in self.K:
            CKSNAP_item = self.get_CKSNAP_item(seq, k)
            CKSNAP.extend(CKSNAP_item)
            # print(f"item_{k}:", CKSNAP)
        return CKSNAP

    def run_fea(self, seq_list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("CKSNAP Params", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            # print("out:", out)
            all_out.extend(out)
            # print("all_out:", all_out)
        return np.array(all_out)

    def get_params(self):
        print("CKSNAP Params", self.__dict__)


if __name__ == '__main__':
    cksnap = CKSNAP()

    run = cksnap.run_fea(["ATGCACGTNCACGT"])
    print(run)
