from __future__ import annotations
from typing import Tuple, List, Generator, Dict

from shapely.geometry import LineString

from MCFS.isolines import IsoLine


class StitchingTuple:

    def __init__(self, Iu:IsoLine, Iv:IsoLine, Iu_p:tuple, Iu_pidx:int, Iv_q:tuple, Iv_qidx:int) -> None:
        self.Iu, self.Iv = Iu, Iv
        self.Iu_p_idx, self.Iu_p = Iu_pidx, Iu_p
        self.Iv_q_idx, self.Iv_q = Iv_qidx, Iv_q
        self.Iu_Bp_idx, self.Iu_Bp = Iu.B(Iu_pidx)
        self.Iv_Bq_idx, self.Iv_Bq = Iv.B(Iv_qidx)
    
    def validate(self) -> bool:
        return not LineString([self.Iu_Bp, self.Iv_Bq]).intersects(LineString([self.Iu_p, self.Iv_q]))

    @property
    def index(self) -> tuple:
        return (self.Iu_p_idx, self.Iv_q_idx)

    def __hash__(self) -> int:
        return self.index.__hash__()

    def __eq__(self, other:StitchingTuple) -> bool:
        return self.index == other.index
    
    def __ne__(self, other:StitchingTuple) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other:StitchingTuple) -> bool:
        # comparing s1, s2 in the same stitching tuple set
        return self.index < other.index

    def get(self, Ix:IsoLine) -> Tuple[int, tuple, int, tuple]:
        if Ix == self.Iu:
            return self.Iu_p_idx, self.Iu_p, self.Iu_Bp_idx, self.Iu_Bp
        else:
            return self.Iv_q_idx, self.Iv_q, self.Iv_Bq_idx, self.Iv_Bq


class StitchingTupleSet:

    def __init__(self, Iu:IsoLine, Iv:IsoLine) -> None:
        self.Iu, self.Iv = Iu, Iv
        self._dat:List[StitchingTuple] = []
        self._inds_set: Dict[IsoLine, set] = {Iu:set(), Iv:set()}
        self._inds_map: Dict[IsoLine, Dict[int, int]] = {Iu:{}, Iv:{}}
        # for two isolines where no valid stitching tuple exists.
        self.shortest_path = None
        
    def __len__(self) -> int:
        return len(self._dat)
    
    @staticmethod
    def build(Iu:IsoLine, Iv:IsoLine, Ou2v:set, Ov2u:set) -> StitchingTupleSet:
        S = StitchingTupleSet(Iu, Iv)
        for item in Ou2v.intersection(Ov2u):
            s = StitchingTuple(Iu, Iv, *item)
            if s.validate():
                S._dat.append(s)
                S._inds_set[Iu].add(s.get(Iu)[0])
                S._inds_set[Iv].add(s.get(Iv)[0])
        
        S._dat.sort()
        S._inds_map[Iu] = {s.get(Iu)[0]:i for i, s in enumerate(S._dat)}
        S._inds_map[Iv] = {s.get(Iv)[0]:i for i, s in enumerate(S._dat)}
        return S
    
    @staticmethod
    def from_raw_data(Iu:IsoLine, Iv:IsoLine, dat:List[StitchingTuple]) -> StitchingTupleSet:
        S = StitchingTupleSet(Iu, Iv)
        S._dat = dat
        S._inds_set[Iu] = set([s.get(Iu)[0] for s in S._dat])
        S._inds_set[Iv] = set([s.get(Iv)[0] for s in S._dat])

        S._dat.sort()
        S._inds_map[Iu] = {s.get(Iu)[0]:i for i, s in enumerate(S._dat)}
        S._inds_map[Iv] = {s.get(Iv)[0]:i for i, s in enumerate(S._dat)}
        return S

    def iterate(self, U:set={}) -> Generator[StitchingTuple]:
        for s in self._dat:
            if (s.Iu_p, s.Iu_Bp) not in U and (s.Iv_q, s.Iv_Bq) not in U:
                yield s
