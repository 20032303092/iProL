import numpy as np
from .ANF import ANF
from .NCP import NCP


class RFHC:
    """
    NCP + ANF
    """

    def __init__(self, ncp_obj: NCP, anf_obj: ANF):
        self.ncp = ncp_obj
        self.anf = anf_obj

    def get_fea(self, seq):
        vec_ncp = self.ncp.get_fea(seq)
        vec_anf = self.anf.get_fea(seq)
        vec_anf = np.expand_dims(vec_anf, axis=-1)
        return np.concatenate((vec_ncp, vec_anf), axis=-1, dtype=np.float32)

    def run_fea(self, seq_list):
        print("RFHC Params", self.__dict__)
        vec_ncp = self.ncp.run_fea(seq_list)
        # print(vec_ncp)
        vec_anf = self.anf.run_fea(seq_list)
        # print(vec_anf)
        vec_anf = np.expand_dims(vec_anf, axis=-1)
        # print(vec_anf)
        return np.concatenate((vec_ncp, vec_anf), axis=-1, dtype=np.float32)

    def get_params(self):
        print("RFHC Params", self.__dict__)


if __name__ == '__main__':
    r = RFHC(NCP(), ANF())
    seq_list = ["ATGC", "AAAA"]
    print(r.get_fea(seq_list[0]))
    vecs = r.run_fea(seq_list)
    print(vecs)
