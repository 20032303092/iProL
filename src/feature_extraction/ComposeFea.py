import numpy as np


class ComposeFea:
    def __init__(self, *arg):
        """
        传入不同的特征对象
        根据传入的顺序，按最后一个维度，进行叠加
        :param arg:
        """
        self.fea_objs = arg

    def get_fea(self, seq):
        vec_list = []
        for obj in self.fea_objs:
            vec_temp = obj.get_fea(seq)
            if len(vec_temp.shape) == 1:
                vec_temp = np.expand_dims(vec_temp, axis=-1)
            vec_list.append(vec_temp)
        # print(vec_list)
        vec = np.concatenate(vec_list, axis=-1)

        return vec

    def run_fea(self, seq_list):
        vec_list = []
        for obj in self.fea_objs:
            vec_temp = obj.run_fea(seq_list)
            if len(vec_temp.shape) == 2:
                vec_temp = np.expand_dims(vec_temp, axis=-1)
            vec_list.append(vec_temp)
        # print(vec_list)
        vec = np.concatenate(vec_list, axis=-1)
        return vec

    def get_params(self):
        print("Compose Feature Params:>>>>>>>>>>>>")
        for obj in self.fea_objs:
            obj.get_params()
        print("End<<<<<<<<<<<<<<<")


from .EIIP import EIIP_trans
from .Onehot import Onehot
from .NCP import NCP
from .ANF import ANF
from .EIIP import EIIP_anf


class OnehotEIIP(ComposeFea):
    """ (B,S,5) """

    def __init__(self, onehot_obj: Onehot, eiip_trans: EIIP_trans):
        super().__init__(onehot_obj, eiip_trans)


class OnehotANF(ComposeFea):
    """ (B,S,5) """

    def __init__(self, onehot_obj: Onehot, anf_obj: ANF):
        super().__init__(onehot_obj, anf_obj)


class OnehotBiANF(ComposeFea):
    """ (B,S,6) """

    def __init__(self, onehot_obj: Onehot, anf_obj_l: ANF, anf_obj_r: ANF):
        anf_obj_l.direction = "left"
        anf_obj_r.direction = "right"
        super(OnehotBiANF, self).__init__(onehot_obj, anf_obj_l, anf_obj_r)


class OnehotEIIP_anf(ComposeFea):
    """ (EIIP*ANF) (B,S,5) """

    def __init__(self, onehot_obj: Onehot, eiip_anf_obj: EIIP_anf):
        super().__init__(onehot_obj, eiip_anf_obj)


class OnehotEIIP_BiAnf(ComposeFea):
    """ (B,S,6) """

    def __init__(self, onehot_obj: Onehot, eiip_anf_obj_l: EIIP_anf, eiip_anf_obj_r: EIIP_anf):
        eiip_anf_obj_l.anf.direction = "left"
        eiip_anf_obj_r.anf.direction = "right"
        super().__init__(onehot_obj, eiip_anf_obj_l, eiip_anf_obj_r)


class NCPEIIP(ComposeFea):
    """ (B,S,4) """

    def __init__(self, ncp_obj: NCP, eiip_trans: EIIP_trans):
        super().__init__(ncp_obj, eiip_trans)


class NCPEIIP_anf(ComposeFea):
    """ (B,S,4) """

    def __init__(self, ncp_obj: NCP, eiip_anf_obj: EIIP_anf):
        super().__init__(ncp_obj, eiip_anf_obj)


class NCPEIIP_BiAnf(ComposeFea):
    """ (B,S,5) """

    def __init__(self, ncp_obj: NCP, eiip_anf_obj_l: EIIP_anf, eiip_anf_obj_r: EIIP_anf):
        eiip_anf_obj_l.anf.direction = "left"
        eiip_anf_obj_r.anf.direction = "right"
        super().__init__(ncp_obj, eiip_anf_obj_l, eiip_anf_obj_r)


class NCPEIIPANF(ComposeFea):
    """ (B,S,5) """

    def __init__(self, ncp_obj: NCP, eiip_trans: EIIP_trans, anf_obj: ANF):
        super(NCPEIIPANF, self).__init__(ncp_obj, eiip_trans, anf_obj)


class NCPEIIPBiANF(ComposeFea):
    """ (B,S,6) """

    def __init__(self, ncp_obj: NCP, eiip_trans: EIIP_trans, anf_obj_l: ANF, anf_obj_r: ANF):
        anf_obj_l.direction = "left"
        anf_obj_r.direction = "right"
        super(NCPEIIPBiANF, self).__init__(ncp_obj, eiip_trans, anf_obj_l, anf_obj_r)


if __name__ == '__main__':
    """ 测试ComposeFea类 实现组合特征 """
    # eiip = EIIP.EIIP(k=[1])
    # onehot = Onehot.Onehot()
    # cf = ComposeFea(eiip, onehot)
    # cf.get_params()
    # vec = cf.get_fea("ACGT")
    # print(vec)
    # seq_list = ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT"]
    # vecs = cf.run_fea(seq_list)
    # print(vecs)
    """ 测试OnehotEIIP组合特征 """
    oe = OnehotEIIP(Onehot(), EIIP_trans())
    print(oe)
    print(oe.get_fea("ATGCA"))
