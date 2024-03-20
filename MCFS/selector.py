from __future__ import annotations
from collections import defaultdict
import numpy as np

from MCFS.utils import curvature
from MCFS.isolines import IsoLine
from MCFS.isograph import IsoGraph
from MCFS.stitching_tuple import StitchingTuple


class Selector:
    
    def __init__(self, p_in:tuple, r:IsoLine) -> None:
        self.CFS_rec = defaultdict(set)
        self.CFS_rec[r].add(p_in)
    
    def select(self, selector_type:str, IG:IsoGraph, Iu:IsoLine, Iv:IsoLine, U:set) -> StitchingTuple:
        if selector_type == "random":
            return Selector.random(IG, Iu, Iv, U)
        elif selector_type == "CFS":
            return self.CFS(IG, Iu, Iv, U)
        elif selector_type == "MCS":
            return Selector.MCS(IG, Iu, Iv, U)
    
    @staticmethod
    def random(IG:IsoGraph, Iu:IsoLine, Iv:IsoLine, U:set):
        S_uv = list(IG[Iu][Iv]["S"].iterate(U))
        return S_uv[np.random.choice(len(S_uv))]

    @staticmethod
    def MCS(IG:IsoGraph, Iu:IsoLine, Iv:IsoLine, U:set):
        min_pair = (float("inf"), None)
        for s in IG[Iu][Iv]["S"].iterate(U):
            delta_k = Selector.diff_stitching_curvature(Iu, Iv, s)
            if delta_k < min_pair[0]:
                min_pair = (delta_k, s)
        if not min_pair[1]:
            return IG[Iu][Iv]["S"]._dat[0]
        return min_pair[1]

    @staticmethod
    def diff_stitching_curvature(Iu:IsoLine, Iv:IsoLine, s:StitchingTuple) -> float:
        su_idx, su, tu_idx, tu = s.get(Iu)
        sv_idx, sv, tv_idx, tv = s.get(Iv)

        if (tu_idx + 1) % (len(Iu) - 1) == su_idx:
            # n2tu -> n_tu -> tu -> su -> n_su -> n2su
            n_tu, n_su = Iu.B(tu_idx)[1], Iu.A(su_idx)[1]
            n2tu, n2su = Iu.B(tu_idx,2)[1], Iu.A(su_idx,2)[1]
        else:
            # n2tu <- n_tu <- tu <- su <- n_su <- n2su
            n_tu, n_su = Iu.A(tu_idx)[1], Iu.B(su_idx)[1]
            n2tu, n2su = Iu.A(tu_idx,2)[1], Iu.B(su_idx,2)[1]

        if (tv_idx + 1) % (len(Iv) - 1) == sv_idx:
            # n2tv -> n_tv -> tv -> sv -> n_sv -> n2sv
            n_tv, n_sv = Iv.B(tv_idx)[1], Iv.A(sv_idx)[1]
            n2tv, n2sv = Iv.B(tv_idx,2)[1], Iv.A(sv_idx,2)[1]
        else:
            # n2tv <- n_tv <- tv <- sv <- n_sv <- n2sv
            n_tv, n_sv = Iv.A(tv_idx)[1], Iv.B(sv_idx)[1]
            n2tv, n2sv = Iv.A(tv_idx,2)[1], Iv.B(sv_idx,2)[1]

        return curvature(np.array([n2su, n_su, su, sv, n_sv, n2sv]))[2:4].mean() + \
               curvature(np.array([n2tu, n_tu, tu, tv, n_tv, n2tv]))[2:4].mean() - \
               curvature(np.array([n2tu, n_tu, tu, su, n_su, n2su]))[2:4].mean() - \
               curvature(np.array([n2tv, n_tv, tv, sv, n_sv, n2sv]))[2:4].mean()
    
    def CFS(self, IG:IsoGraph, Iu:IsoLine, Iv:IsoLine, U:set):
        ret = None
        for s in IG[Iu][Iv]["S"].iterate(U):
            su_idx, su, tu_idx, tu = s.get(Iu)
            sv_idx, sv, tv_idx, tv = s.get(Iv)
            if not ret:
                ret = s
            elif tu in self.CFS_rec[Iu] or tv in self.CFS_rec[Iv]:
                ret = s
                break

        self.CFS_rec[Iv].add(sv)
        return ret

