from itertools import product

import numpy as np
import math

from joblib import Parallel, delayed

from .physicalChemical import PhysicalChemical, PhysicalChemicalType


class PseKNC_I:
    """
    包含(4k+λ)个分量的载体的DNA序列
    对于不确定碱基N，直接将其物化性质置0
    :return
    PseKNC1: du先normalization，再计算W*tao，最后整体normalization
    PseKNC2： du先normalization，再计算tao并normalization，最后依据W设置du、tao之间的比例
    """

    def __init__(self, seq="ATGCACGCAT", K=4, lam=5, W=0.1,
                 pcf_array=["旋转扭曲", "左右倾斜", "前后卷动", "左右移动", "前后移动", "上下移动"],
                 n=2):
        # physicochemical feature value

        # define param
        """
        K: k-tuple
        w: the weight between local sequence and global sequence
        lam: λ, 层级关系，即相隔λ长度之间的k-tuple核苷酸的关系
        pcf_array: 具体的特征名词数组
        pcf：∧, number of physicochemical feature
        n: 选择几元性质
        """
        self.K = K
        self.W = W
        self.lam = lam
        self.pcf_array = pcf_array
        self.pcf = len(self.pcf_array)
        self.n = n

        # other param
        """
        du_len: the length of k-tuple occurrence frequency
        tao_len: 长程序列效应的长度（每种层级关系遍历一次所有的物化性质）
        PseKNC: the final output
        L: the length of DNA sequence
        L_hat <= L - K
        """
        self.du_len = int(math.pow(4, self.K))
        self.tao_len = self.lam * self.pcf
        self.PseKNC = np.zeros(((self.du_len + self.tao_len)))
        # print(self.PseKNC.shape)
        seq = seq
        self.L = len(seq)


class PseKNC_II:
    """
    对于不确定碱基N，直接将其物化性质置0
    :return
    PseKNC1: du先normalization，再计算W*tao，最后整体normalization
    PseKNC2： du先normalization，再计算tao并normalization，最后依据W设置du、tao之间的比例
    D_Pseknc_1 = [d~1~,d~2~,d~3~, ... ,d~(4^K^)~, τ~1~,τ~2~, ... , τ~λ~ ,τ~λ+1~, ... ,τ~λΛ~] 【**这个是没有标准化的形式，大致这么理解**】
    """

    def __init__(self, set_pc_list, k_tuple, lam=5, W=0.5, n_pc=None, seq_type="DNA", n_jobs=1):

        # define param
        """
        K: k-tuple
        w: the weight between local sequence and global sequence
        lam: λ, 层级关系，即相隔λ长度之间的k-tuple核苷酸的关系
        set_pc_list: needed physicochemical names list
        pc_dict：physicochemical table about all physicochemical values
        Lambda：∧, number of physicochemical feature
        n_pc: 选择几元物化性质
        """
        self.k_tuple = k_tuple
        self.W = W
        self.lam = lam
        self.n_jobs = n_jobs

        self.Lambda = len(set_pc_list)
        self.seq_type = seq_type
        if n_pc is None and self.k_tuple <= 3:
            self.n_pc = self.k_tuple
        elif n_pc == 2 or n_pc == 3:
            self.n_pc = n_pc
        else:
            raise ValueError(f"n_pc = {self.n_pc}, n_pc value is error")
        self.set_pc_list = set_pc_list
        self.pc_dict = self._get_pc_dict()

        # other param
        """
        du_len: the length of k-tuple occurrence frequency
        tau_len: 长程序列效应的长度（每种层级关系遍历一次所有的物化性质）
        PseKNC: the final output
        """
        self.du_len = int(math.pow(4, self.k_tuple))
        self.tau_len = self.lam * self.Lambda

    def __call__(self, seq_list: list, case_type="paper_case"):

        PseKNC_feature = self.run_fea(seq_list=seq_list, case_type=case_type)

        return PseKNC_feature

    def get_fea(self, seq, case_type="paper_case"):
        """
        if case_type == "paper_case":
            du先normalization得到k-tuple核苷酸出现频率，这一部分的值的和为1；
            得到tao原始值后，再整体normalization
            即PseKNC数组的所有值的和是1
        else:
             du和tao[分开独自]normalization，再依据W统一normalization
            即PseKNC数组的所有值的和是2
            这里W表示du和tao两类占的比例
        normalization: item / sum
        :param seq:
        :param case_type:
        :return:
        """
        if self.lam >= (len(seq) - self.k_tuple):
            raise ValueError(f"lam = {self.lam} is greater than the [length of seq - k_tuple] !")
        if case_type == "paper_case":
            feature_vector = self.getPseKNC_II_paper_case(seq)
        else:
            feature_vector = self.getPseKNC_II_case1(seq)
        return np.array(feature_vector)

    def run_fea(self, seq_list: list, case_type="paper_case"):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("PseKNC_II params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq, case_type=case_type)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    # 第一步：计算pow(4,K)个k-tuple元组
    def _get_du_part_before_normalization(self, seq):
        """
        得到k-tuple核苷酸normalization出现频率
        :param seq:
        :return:
        """
        PseKNC_part1 = np.zeros(self.du_len)
        K_tuple_array = self._get_K_tupleCount(seq)
        # print("K_tuple_array.sum():", K_tuple_array.sum())
        for index, item in enumerate(K_tuple_array, 0):
            value = item / K_tuple_array.sum()
            PseKNC_part1[index] = value
        # print(PseKNC_part1)
        return PseKNC_part1

    def _get_K_tupleCount(self, seq):
        """
        用于对k-tuple计数，A:0, T:1, G:2, C:3,
        if k_tuple-3, then按照AAA，AAT，AAG，...，TTT排序,真实坐标即是4进制表示
        如果含N，则抛弃
        :return:
        """
        seq_len = len(seq)
        K_tuple_array = np.zeros(self.du_len)
        # print(K_tuple_array.shape)
        k_tuple_count = 0
        for j in range(seq_len - self.k_tuple + 1):
            tip = seq[j: j + self.k_tuple]
            # print("tip:", tip)
            if tip.find('N') != -1:
                # print("遇到N")
                continue
            tip = tip.replace('A', '0')
            tip = tip.replace('C', '1')
            tip = tip.replace('G', '2')
            tip = tip.replace('T', '3')
            # print("tip_value_4:", tip)
            index = self._get_K_Tuple_index(tip)
            # print("index:", index)
            K_tuple_array[int(index)] += 1
            k_tuple_count += 1
        # print("k_tuple_count:", k_tuple_count)
        return K_tuple_array

    def _get_K_Tuple_index(self, tip: str):
        """
        4进制转换成10进制，得到该k-tuple所在的下标
        :param tip: like AGC ATC ... , if k_tuple=3
        :return:
        """
        index = 0
        tip_len = len(tip)
        for item in tip:
            index += int(item) * math.pow(4, tip_len - 1)
            tip_len -= 1
        return index

    # 第二步：计算λΛ部分
    def _get_tau_part_case1_before_normalization(self, seq):
        """
        tao单独normalization
        :param tauArray:
        :return:
        """
        taoArray = self._get_tauArray_before_normalization(seq)
        PseKNC_part2 = np.zeros(self.tau_len)
        for index, tau_item in enumerate(taoArray):
            PseKNC_part2[index] = tau_item / taoArray.sum()
        return PseKNC_part2

    def _get_tau_part_case2_before_normalization(self, seq):
        """
        :param taoArray:
        :return:
        """
        taoArray = self._get_tauArray_before_normalization(seq)
        PseKNC_part2 = np.zeros(self.tau_len)
        for index, tao_item in enumerate(taoArray):
            PseKNC_part2[index] = self.W * tao_item
        return PseKNC_part2

    def _get_PCValue(self, R_tip_left, R_tip_right, physicalchemical_name):
        """
        计算两相隔m层级远的n元核苷酸的某一物理性质的值
        :param physicalchemical_name:
        :param R_tip_left:
        :param R_tip_right:
        :return:
        """

        try:
            PCValue = self.pc_dict[physicalchemical_name][R_tip_left] * self.pc_dict[physicalchemical_name][R_tip_right]
            # print(physicalchemical_name, R_tip_left, pc_dict[physicalchemical_name][R_tip_left], R_tip_right,
            #       pc_dict[physicalchemical_name][R_tip_right])
        except KeyError:
            # print(R_tip_left, R_tip_right, physicalchemical_name, "has KeyError, set PCValue = 0")
            PCValue = 0
        return PCValue

    def _get_sum_of_J(self, seq, tier, physicalchemical_name):
        """
        求J_len个J的值,get tau value
        :param tier:层级，从0计数
        :param feature_name:特征名称
        :return:
        """

        # J_len: 每一层级每种物理化学性质(per tau) J的个数
        J_len = len(seq) - self.k_tuple - tier - 1
        sum_of_J_value = 0
        m = tier + 1
        for i in range(J_len):
            R_tip_left = seq[i]
            R_tip_right = seq[i + m]
            for r in range(1, self.n_pc):
                R_tip_left += seq[i + r]
                R_tip_right += seq[i + m + r]
            # print(R_tip_left, R_tip_right)
            sum_of_J_value += self._get_PCValue(R_tip_left, R_tip_right, physicalchemical_name)
        # print(sum_of_J_value)
        return sum_of_J_value

    def _get_tauArray_before_normalization(self, seq):
        """
        得到τ的np数组的原始数据，即未经过normalization
        :return:
        """
        #  tier:层级，从0计数
        seq_len = len(seq)
        taoArray = np.zeros(self.tau_len)
        for tier in range(self.lam):
            # print("tier:", tier, "=================================================")
            for pcf_index in range(self.Lambda):
                # print(tier * self.pcf + pcf_index)
                taoArray[tier * self.Lambda + pcf_index] = 1 / (seq_len - self.k_tuple - tier - 1) * self._get_sum_of_J(
                    seq, tier, self.set_pc_list[pcf_index])
        return taoArray

    # 第三步：得到最后的PseKNC
    def getPseKNC_II_case1(self, seq):
        """
        du和tao[分开独自]normalization，再依据W统一normalization
        即PseKNC数组的所有值的和是2
        这里W表示du和tao两类占的比例
        :return:
        """
        PseKNC_part1 = self._get_du_part_before_normalization(seq)
        PseKNC_part2 = self._get_tau_part_case1_before_normalization(seq)
        PseKNC_vector = np.append(PseKNC_part1, PseKNC_part2)
        # 统一归一化
        # Sum = PseKNC_vector.sum()
        # print("case1 sum:", Sum)
        # W*1 + （1-W）*1；两边都归一化了；sum一定是一
        for index, value in enumerate(PseKNC_vector, 0):
            # PseKNC_vector[index] = value / Sum
            if index < self.du_len:
                PseKNC_vector[index] = value * (1 - self.W)
            else:
                PseKNC_vector[index] = value * self.W
        return PseKNC_vector

    # 论文中的是这种的
    def getPseKNC_II_paper_case(self, seq):
        """
        2014_Analytical Biochemistry_PseKNC - A flexible web server for generating pseudo K-tuple nucleotide composition
        2018_iTerm-PseKNC - a sequence-based tool for predicting bacterial transcriptional terminators
        du先normalization得到k-tuple核苷酸出现频率，这一部分的值的和为1；
        得到tao原始值后，再整体normalization
        即PseKNC数组的所有值的和是1
        :return:
        """
        PseKNC_part1 = self._get_du_part_before_normalization(seq)
        PseKNC_part2 = self._get_tau_part_case2_before_normalization(seq)
        PseKNC_vector = np.append(PseKNC_part1, PseKNC_part2)
        # 统一归一化
        Sum = PseKNC_vector.sum()
        for index, value in enumerate(PseKNC_vector, 0):
            PseKNC_vector[index] = value / Sum
        return PseKNC_vector

    def _get_pc_dict(self):
        pc_dict = None
        if self.n_pc == 2:
            if self.seq_type == "DNA":
                pc_dict = PhysicalChemical(PhysicalChemicalType.DiDNA_standardized).pc_dict
            elif self.seq_type == "RNA":
                pc_dict = PhysicalChemical(PhysicalChemicalType.DiRNA_standardized).pc_dict
        elif self.n_pc == 3:
            if self.seq_type == "DNA":
                pc_dict = PhysicalChemical(PhysicalChemicalType.TriDNA_standardized).pc_dict
            elif self.seq_type == "RNA":
                # todo: TriRNA
                pc_dict = PhysicalChemical(PhysicalChemicalType.TriDNA_standardized).pc_dict
        else:
            raise ValueError(f"n_pc = {self.n_pc}, check n_pc value!!!")

        return pc_dict

    def get_params(self):
        print("PseKNC_II Params", self.__dict__)

class PseDNC_II(PseKNC_II):
    def __init__(self, set_pc_list, lam=5, W=0.5, n_pc=None, seq_type="DNA", n_jobs=1):
        super(PseDNC_II, self).__init__(set_pc_list=set_pc_list, k_tuple=2, lam=lam, W=W, n_pc=n_pc, seq_type=seq_type,
                                        n_jobs=n_jobs)

    def get_PseDNC_II(self, seq, case_type="paper_case"):
        return PseKNC_II.get_fea(self, seq, case_type)

    def run_PseDNC_II(self, seq_list: list, case_type="paper_case"):
        PseDNC_II_feature = PseKNC_II.run_fea(self, seq_list, case_type)
        return PseDNC_II_feature

    def get_params(self):
        print("PseDNC_II Params", self.__dict__)

class PseTNC_II(PseKNC_II):
    def __init__(self, set_pc_list, lam=5, W=0.5, n_pc=None, seq_type="DNA", n_jobs=1):
        super(PseTNC_II, self).__init__(set_pc_list=set_pc_list, k_tuple=3, lam=lam, W=W, n_pc=n_pc, seq_type=seq_type,
                                        n_jobs=n_jobs)

    def get_PseTNC_II(self, seq, case_type="paper_case"):
        return PseKNC_II.get_fea(self, seq, case_type)

    def run_PseTNC_II(self, seq_list: list, case_type="paper_case"):
        PseDNC_II_feature = PseKNC_II.run_fea(self, seq_list, case_type)
        return PseDNC_II_feature

    def get_params(self):
        print("PseTNC_II Params", self.__dict__)

if __name__ == '__main__':
    set_pc_list = [
        "Bendability-DNAse",
        "Bendability-consensus",
        "Trinucleotide GC Content",
        "Nucleosome positioning",
        "Consensus_roll",
        "Consensus_roll",
        "Consensus_Rigid",
        "Consensus_Rigid",
    ]
    set_pc_list = [
        "Twist",
        "Tilt",
        "Roll",
        "Shift",
        "Slide",
        "Rise"
    ]
    seq_list = ["ATGCATCATC", "ATGCATGNGCATC", "AGCTGGNATGCA", "GGAGGAGCACCGGT", "AATACAGGCGGAGCAGCAGA",
                "CACAGGCTCTTAAAGA", "CCCAGATGGAGGTG", "CAGGGAGGAGTTTGG", "TGCATGNGCAT", "CCAGATGGAGGTGGGGA",
                "AGGAGTTTGGAGAG", "GGCTCTTAAAG", "CGCTTCCCACACTCGCCG", "ATGCATCATNC", "ATGCATNGNGCATC", "AGCTGGNNATGCA",
                "GGNAGGAGNCACCGGT", "AATACAGGNCGGAGCAGCAGA", "CTTCCCACACTCGCCG", "CTTCCCACACTCGCCAAG",
                "CTTCTTCCACACTCGCCG", "AGGNCGGAGCAGCA", "GGGAGGNCGGAGCAGCA", "AGGNCGGAGCAGCATTT", "AGGNCGGAGCAGCAGTA",
                "AGGNCGGAGCAGCAGCC", "TCCCACACTCGNN", "TCCCACACTCGNTGA", "TCCCACACTCGNA", "NTGATCCCACACTCG",
                "TCCCACACTCGTGGGG", "TCCCACACTCGAAG", "GAGGNNAGTTAAG", "GAGGNNAGTTTGN", "NTGGAGGNNAGTT",
                "AGCTGAGGNNAGTT", "AGGGAGGAGTTTGGA", "ATGGAGGTGGGG", "AGAGGGNGAAGG", "CTTCCNCACACTCGC",
                "CTTCCNCANCACTNCGC",
                "CACANGGCTCTNTAAAGA", "CCCAGATGGNAGGTG", "CAGGGAGGNNAGTTTGG", "TGCNATGNGCAT", "CCAGATGNGAGGTGGGGA",
                "AGGAGTTTGNGAGAG", "GGCTCTNTAAAG", "CGCTTCCNCACACTCGCCG",
                "AAAGAGGGNGAAGGGG", "CCCAGATGGANGGTGGGGAGG",
                "GGACCAGGGAGGAGTTTGGAGANGAA"
                "AAAGAGGGGAAGGGG", "CCCAGATGGAGGTGGGGAGG", "GGACCAGGGAGGAGTTTGGAGAGAA"]
    pseknc = PseKNC_II(set_pc_list, k_tuple=2)
    PseKNC_feature = pseknc(seq_list, case_type="paper_case")

    psednc = PseDNC_II(set_pc_list, n_jobs=5)

    psednc_feature = psednc(seq_list, case_type="paper_case")

    print(PseKNC_feature - psednc_feature)
