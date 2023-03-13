from collections.abc import Iterable
from itertools import product

import numpy as np
from joblib import Parallel, delayed
import math


class Kmer:
    """
    Kmer：Kmer描述给定序列中k个相邻核酸的出现频率。
    可以将k=1，2，3，4，5，...，n等的特征向量拼接起来作为一个组合的特征向量使用,还可以设置步长step
    考虑**核酸组成(NAC)、二核酸组成(DNC)、三核酸组成(TNC)和四核酸组成(TeNC)**。
    NAC、DNC、TNC、TENC分别生成4D、16D、64D和256D特征向量。随后，将这四个合成的特征向量连接起来，得到340D的特征向量。
    format 'vec':
    [N(A)/K1,N(C)/K1,N(G)/K1,N(T)/K1,N(AA)/K2,...,N(TT)/K2,N(AAA)/K3,...]
    freq = N /K

    format 'idx':
    seq --> [12, 24,...]
    format 'seq':
    seq --> ['ATC', 'TCN',...]
    """

    def __init__(self, k=range(1, 5), stride=1, return_type="vec", n_jobs=1):
        """
        kmer_base
        :param k:
        :param stride: ATGACGT: A...,T...,G...,A...,... when stride=1
        :param return_type: 'vec','idx','seq'
        :param n_jobs:
        """
        # stride=1
        self.k = k
        self.n_jobs = n_jobs
        self.stride = stride
        self.return_type = return_type
        if return_type in ["idx", 'seq']:
            if isinstance(self.k, Iterable):
                if len(self.k) != 1:
                    raise ValueError("the length of `k` must be 1")
            elif isinstance(self.k, int):
                pass
            else:
                raise ValueError("`k` must be int when `return_type` is idx or seq!")

    def get_params(self):
        print("Kmer Params:", self.__dict__)

    # 提取核苷酸类型（排列组合）
    def nucleotide_type(self, k):
        z = []
        for i in product('ACGT', repeat=k):  # 笛卡尔积（有放回抽样排列）
            # print(i)
            z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
        # print(z)
        return z

    def _get_Kmer_item(self, seq, k_item):
        N = (len(seq) - k_item) // self.stride + 1
        atgc_list = self.nucleotide_type(k_item)
        freq_atgc_dict = dict(zip(atgc_list, np.zeros(len(atgc_list))))
        kmer_idx_dict = dict(zip(atgc_list, range(int(math.pow(4, k_item)))))
        kmer_idx_list = []
        kmer_seq = ""
        for idx in range(0, len(seq) - k_item + 1, self.stride):
            # ((seq_l - kmer) // stride + 1)
            atgc = seq[idx:idx + k_item]
            kmer_seq += (atgc + ' ')
            # print(k_item, " | ", atgc)
            try:
                freq_atgc_dict[atgc] += 1
            except KeyError:
                N -= 1
            try:
                kmer_idx_list.append(kmer_idx_dict[atgc])
            except KeyError:
                kmer_idx_list.append(4 ** k_item)
        # print(k_item, " | ", freq_atgc_dict)
        for k, v in freq_atgc_dict.items():
            freq_atgc_dict[k] = v / N
        # print(k_item, " | ", freq_atgc_dict)
        kmer_vec_np = np.array(list(freq_atgc_dict.values()))

        return kmer_vec_np, kmer_seq[:-1], kmer_idx_list

    def get_fea(self, seq, return_type):
        kmer_vec = []
        if isinstance(self.k, int):
            kmer_vec, kmer_seq_list, kmer_idx_list = self._get_Kmer_item(seq, self.k)
        else:
            for k_item in self.k:
                kmer_vec_np, kmer_seq_list, kmer_idx_list = self._get_Kmer_item(seq, k_item)
                # print(kmer_item_vec)
                kmer_vec.extend(kmer_vec_np)
        if return_type == 'idx':
            return kmer_idx_list
        elif return_type == 'seq':
            return kmer_seq_list
        return kmer_vec

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq, return_type=self.return_type)
                           for seq in seq_list
                           )
            all_out.extend(out)
            if self.return_type == 'idx':
                return all_out
            elif self.return_type == 'seq':
                return all_out
        return np.array(all_out)


if __name__ == '__main__':
    kmer = Kmer(k=[2], stride=1, return_type='idx')
    kmer_vec = kmer.run_fea(["AACTNACGT", "AGCTNACGT"])
    print("shape:", len(kmer_vec))
    print(kmer_vec)
