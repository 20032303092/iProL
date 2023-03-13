from itertools import product

import numpy as np
from joblib import Parallel, delayed


class EIIP:
    """
    这个是结合kmer频率算出的
    显示自由电子能量分布，{A，C，G，T}值为{0.1260，0.1340，0.0806，0.1335}。
    S = [EIIP~AAA~f~AAA~, ... , EIIP~TTT~f~TTT~]
    其中EIIP~pqr~=EIIP~p~+EIIP~q~+EIIP~r~表示三核苷酸(pqr)EIIP值之一，并且pqr∈{G，A，C，T}； f~pqr~表示三核苷酸(pqr)的频率。
    最终，EIIP提供了64D特征向量。
    """
    EIIP_value_dict = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335}

    def __init__(self, k=range(1, 5), n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs

    # 提取核苷酸类型（排列组合）
    def nucleotide_type(self, k_item: int):
        z = []
        for i in product('ACGT', repeat=k_item):  # 笛卡尔积（有放回抽样排列）
            # print(i)
            z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
        # print(z)
        return z

    def _get_EIIP_freq(self, seq, k_item):
        N = len(seq) - k_item + 1
        atgc_list = self.nucleotide_type(k_item)
        freq_atgc_dict = dict(zip(atgc_list, np.zeros(len(atgc_list))))
        for idx in range(0, len(seq) - k_item + 1):
            atgc = seq[idx:idx + k_item]
            # print(k_item, " | ", atgc)
            try:
                freq_atgc_dict[atgc] += 1
            except KeyError:
                N -= 1
        # print(k_item, " | ", freq_atgc_dict)
        for k, v in freq_atgc_dict.items():
            freq_atgc_dict[k] = v / N
        # print(k_item, " | ", freq_atgc_dict)
        return np.array(list(freq_atgc_dict.values()))

    def _get_EIIP_value(self, k_item):
        EIIP_dict = {}
        for i in product('ACGT', repeat=k_item):
            v = 0
            for s in i:
                v += self.EIIP_value_dict[s]
            k = ''.join(i)  # 把('A,A,A')转变成（AAA）形式
            EIIP_dict.update({k: v})
        # print(EIIP_dict)
        return np.array(list(EIIP_dict.values()))

    def get_fea(self, seq):
        feature = []
        for k_item in self.k:
            EIIP_freq = self._get_EIIP_freq(seq, k_item)
            EIIP_value = self._get_EIIP_value(k_item)
            # print(EIIP_freq)
            # print(EIIP_value)
            feature_item = [freq * value for freq, value in zip(EIIP_freq, EIIP_value)]
            # print(feature_item)
            feature.extend(feature_item)
        return np.array(feature)

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("EIIP Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("EIIP Params", self.__dict__)


class EIIP_trans:
    EIIP_value_dict = {"a": 0.1260, "c": 0.1340, "g": 0.0806, "t": 0.1335}

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        seq_one = []
        for s in seq:
            seq_one.append(self.EIIP_value_dict[str(s).lower()])
        eiip_vec = np.array(seq_one)
        return eiip_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("EIIP(value) Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("EIIP(value) Params", self.__dict__)


from .ANF import ANF


class EIIP_anf(EIIP_trans):
    """
    采用ANF作为基础频率，并乘以当前碱基的自由能
    """

    # EIIP_value_dict = {"a": 0.1260, "c": 0.1340, "g": 0.0806, "t": 0.1335}

    def __init__(self, anf_obj: ANF, n_jobs=1):
        super().__init__(n_jobs)
        self.anf = anf_obj

    def get_fea(self, seq):
        # seq_one = []
        # for s in seq:
        #     seq_one.append(self.EIIP_value_dict[str(s).lower()])
        # eiip_vec = np.array(seq_one)

        eiip_vec = super().get_fea(seq)
        anf_vec = self.anf.get_fea(seq)
        return eiip_vec * anf_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("EIIP(ANF) Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("EIIP(ANF) Params", self.__dict__)


if __name__ == '__main__':
    """ kmer frequency"""
    # eiip = EIIP([3])
    # print(eiip.n_jobs)
    # print(eiip.nucleotide_type(3))
    # feature = eiip.get_fea("GATAATCCCATGGGGCTGTACGCGATTTATATTGGCAGGTTGTATGCCATCCATGGTACCAATGCCAATTTTGGTATTGGG")
    # print(len(feature))
    # print(feature)
    # feature = eiip.run_fea(["GATAATCCCATGGGGCTGTACGCGATTTATATTGGCAGGTTGTATGCCATCCATGGTACCAATGCCAATTTTGGTATTGGG"])
    # print(len(feature))
    # print(feature)

    """ anf frequency"""
    eiip = EIIP_anf(ANF())
    print(eiip.get_fea("ACCGTA"))
    eiip = EIIP_anf(ANF(direction='right'))
    print(eiip.get_fea("ACCGTA"))
