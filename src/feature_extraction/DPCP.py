from itertools import product

import numpy as np
from joblib import Parallel, delayed

from .physicalChemical import PhysicalChemical, PhysicalChemicalType


class DPCP:
    """
    DPCP(a) = f(a)xPCP(Y~a~)b
    Y~a~是第b(b=1，2，…，21)个DPCPs的值。21个DPCP产生336D向量（336=21x16，21种理化性质，16种二元组核苷酸）。
    - 21：21种二核苷酸理化性质
    - 16：16种二元组核苷酸
    - a：二元组核苷酸
    - b：理化性质
    - f(a)：二元组核苷酸频率

    即,格式为:
    [PC_1_AA x f(AA),PC_1_AC x f(AC), ... , PC_1_TT x f(TT), ... , PC_n_AA x f(AA), ... , PC_n_TT x f(TT)]
    [N_sample, n x 16]
    """

    def __init__(self, kmer, set_pc_list, seq_type="DNA", n_jobs=1):
        self.n_jobs = n_jobs
        self.kmer = kmer
        self.seq_type = seq_type
        self.set_pc_list = set_pc_list
        self.pc_dict = self.get_pc_dict()

    def get_pc_dict(self):
        pc_dict = PhysicalChemical(PhysicalChemicalType.DiDNA_standardized).pc_dict
        if self.seq_type == "RNA":
            pc_dict = PhysicalChemical(PhysicalChemicalType.DiRNA_standardized).pc_dict
        if self.kmer == 3:
            pc_dict = PhysicalChemical(PhysicalChemicalType.TriDNA_standardized).pc_dict
        return pc_dict

    # 提取核苷酸类型（排列组合）
    def nucleotide_type(self):
        z = []
        for i in product('ACGT', repeat=self.kmer):  # 笛卡尔积（有放回抽样排列）
            # print(i)
            z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
        # print(z)
        return z

    def get_fea(self, seq: str):
        base_pair_list = self.nucleotide_type()
        freq_base_dict = dict(zip(base_pair_list, np.zeros(len(base_pair_list))))
        N = len(seq) - self.kmer + 1

        for i in range(0, len(seq) - self.kmer + 1):
            kmer_seq = seq[i:i + self.kmer]
            # print(kmer_seq)
            try:
                freq_base_dict[kmer_seq] += 1
            except KeyError:
                N -= 1
        for k, v in freq_base_dict.items():
            freq_base_dict[k] = v / N
        feature = []
        for item in product(self.set_pc_list, base_pair_list):
            # print(item)
            value = self.pc_dict[item[0]][item[1]] * freq_base_dict[item[1]]
            feature.append(value)
        # print(feature)
        return np.array(feature)

    def run_fea(self, seq_list: list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("DPCP Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("DPCP Params", self.__dict__)


if __name__ == '__main__':
    pc_dict = PhysicalChemical(PhysicalChemicalType.DiDNA_standardized).pc_dict

    set_pc_list = ["Base stacking", "Protein induced deformability", "B-DNA twist", "A-philicity", "Propeller twist",
                   "Duplex stability (freeenergy)", "Duplex stability (disruptenergy)", "DNA denaturation",
                   "Bending stiffness", "Protein DNA twist", "Stabilising energy of Z-DNA", "Aida_BA_transition",
                   "Breslauer_dG", "Breslauer_dH", "Breslauer_dS", "Electron_interaction", "Hartman_trans_free_energy",
                   "Helix-Coil_transition", "Ivanov_BA_transition", "Lisser_BZ_transition", "Polar_interaction"]
    dpcp = DPCP(2, set_pc_list, pc_dict, n_jobs=4)
    v = dpcp.get_fea("ATGCANC")
    print(v)
    feature1 = dpcp.run_fea(
        ["ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC",
         "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC""ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC",
         "ATGCANC", "ATTACANC""ATGCANC", "ATTACANC"])
    print(feature1)
    feature = dpcp.run_fea(
        ["ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC",
         "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC",
         "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC",
         "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", "ATTACANC", "ATGCANC", ])
    print(feature)
