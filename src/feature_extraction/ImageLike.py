import numpy as np
from joblib import Parallel, delayed

from .ComposeFea import NCPEIIP, NCPEIIP_anf
from .ANF import ANF
from .EIIP import EIIP_trans
from .Onehot import Onehot


class ImageLike:
    """
    类图像特征，BCHW->(B,3,L,4)
    NCP + EIIP
    ANF + EIIP*ANF  （左右两个方向）
    onehot
    """

    def __init__(self, ne_obj: NCPEIIP, anf_obj: ANF,
                 e_t_obj: EIIP_trans,
                 oh_obj: Onehot, n_jobs=1):
        self.ne_obj = ne_obj
        self.anf_obj = anf_obj
        self.e_t_obj = e_t_obj
        self.oh_obj = oh_obj
        self.n_jobs = n_jobs

    def get_fea(self, seq):
        # NCP + EIIP
        c1_vec = self.ne_obj.get_fea(seq)
        # ANF + EIIP*ANF  （左右两个方向）
        self.anf_obj.direction = 'left'
        c2_0_vec = self.anf_obj.get_fea(seq)

        self.anf_obj.direction = 'right'
        c2_1_vec = self.anf_obj.get_fea(seq)
        et_vec = self.e_t_obj.get_fea(seq)
        c2_2_vec = c2_0_vec * et_vec
        c2_3_vec = c2_1_vec * et_vec
        c2_vec = np.vstack((c2_0_vec, c2_1_vec, c2_2_vec, c2_3_vec)).T
        # onehot
        c3_vec = self.oh_obj.get_fea(seq)

        # vec = np.hstack((c1_vec, c2_vec, c3_vec)).reshape((len(seq), 4, 3))
        vec = np.array((c1_vec, c2_vec, c3_vec))
        # print(c1_vec)
        # print(c2_vec)
        # print(c3_vec)
        return vec

    def run_fea(self, seq_list):
        parallel = Parallel(n_jobs=self.n_jobs)
        print("ImageLike Params:", self.__dict__)
        with parallel:
            all_out = []
            out = parallel(delayed(self.get_fea)
                           (seq=seq)
                           for seq in seq_list
                           )
            all_out.extend(out)
        return np.array(all_out)

    def get_params(self):
        print("ImageLike Params", self.__dict__)


if __name__ == '__main__':
    from NCP import NCP
    import torch

    img = ImageLike(NCPEIIP(NCP(), EIIP_trans()), ANF(), EIIP_trans(), Onehot())
    # vec = img.get_fea("AACCGGTTAA")
    # print(vec)
    seq_list = ['AACGTGCATC', 'GCATGACGAA', 'ACCGTGTAAC']
    vec = img.run_fea(seq_list)
    vec = torch.tensor(vec)
    print(vec)
    print(vec.shape)
