from __future__ import annotations
from typing import List, Tuple
from collections import defaultdict
from itertools import combinations
import heapq

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from MCFS.utils import path_length
from MCFS.isograph import IsoGraph
from MCFS.isolines import IsoLine
from MCFS.planning import unified_CFS
from MCFS.stitching_tuple import StitchingTuple, StitchingTupleSet


def pairwise_isovertices_splitting(
    IG:IsoGraph, T_list:List[IsoGraph], Iu:IsoLine, Iv:IsoLine, pl_dist:dict
) -> Tuple[float, List[StitchingTuple], list, float]:

    S_uv: StitchingTupleSet = IG[Iu][Iv]["S"]
    rec, w, k, L = [], [], len(T_list), len(S_uv)

    print(f" -> PIS: {Iu.index}(size={len(Iu)}), {Iv.index}(size={len(Iv)}), S-size={L}")

    for idx, T in enumerate(T_list):
        rec.append([(Iu, In, T[Iu][In]["S"]) for In in T.neighbors(Iu) if In != Iv])
        w.append(sum([T.nodes[v]["weight"] for v in T.nodes if v != Iu and v != Iv]))
        if Iv in T.nodes:
            rec[-1] += [(Iv, In, T[Iv][In]["S"]) for In in T.neighbors(Iv) if In != Iu]

    len_Iu_pl, len_Iv_pl, itsec_rec = {}, {}, {}
    for s in IG[Iu][Iv]["S"].iterate():
        for t in IG[Iu][Iv]["S"].iterate():
            len_Iu_pl[(s,t)] = len(IsoLine.segment(Iu, s.get(Iu)[0], t.get(Iu)[2]))
            len_Iv_pl[(s,t)] = len(IsoLine.segment(Iv, s.get(Iv)[0], t.get(Iv)[2]))

    def __validate(s_inds:List[int]) -> list:
        nonadj_dist = [0 for _ in rec]
        for idx, item in enumerate(rec):
            si, sj = S_uv._dat[s_inds[idx]], S_uv._dat[s_inds[(idx+1)%k]]
            for Ix, In, _S in item:
                st_idx, ed_idx = si.get(Ix)[0], sj.get(Ix)[0]
                if (Ix.index, st_idx, ed_idx) not in itsec_rec:
                    itsec_rec[(Ix.index, st_idx, ed_idx)] = intersects(_S._inds_set[Ix], si.get(Ix)[0], sj.get(Ix)[0])
                if not itsec_rec[(Ix.index, st_idx, ed_idx)]:
                    if (Ix, In) not in pl_dist:
                        pl_dist[(Ix, In)] = pl_dist[(In, Ix)] = Ix.line.distance(In.line)
                    nonadj_dist[idx] = pl_dist[(Ix, In)]
        
        return nonadj_dist
    
    def __subroutine(i:int) -> Tuple[float, list, int]:
        if i == k-1:
            cnt = 0
            for _i in range(k):
                if s_inds[_i] >= s_inds[(_i+1)%k] + 1:
                    cnt += 1
                if cnt > 1:
                    return (float('inf'), []), (float('inf'), [])

            s_list = [S_uv._dat[s_idx] for s_idx in s_inds]
            nonadj_dist = __validate(s_inds)
            weights = np.zeros(k)
            for idx in range(k):
                si, sj = s_list[idx], s_list[(idx+1)%k]
                weights[idx] = len_Iu_pl[(si,sj)] + len_Iv_pl[(si,sj)] + w[idx]
            if sum(nonadj_dist) == 0:
                return (np.std(weights), s_list), (float('inf'), [])
            else:
                weights += np.array(nonadj_dist)
                return (float('inf'), []), (np.std(weights), s_list)

        _best, _best_invalid = (float('inf'), []), (float('inf'), [])
        while s_inds[i+1] != s_inds[i]:
            _local_best, _local_best_invalid = __subroutine(i+1)
            if _local_best[0] < _best[0]:
                _best = _local_best
            if _local_best_invalid[0] < _best_invalid[0]:
                _best_invalid = _local_best_invalid
            s_inds[i+1] = (s_inds[i+1] + 1) % L
        
        return _best, _best_invalid
    
    cur_iter = 0
    best, best_invalid = (float('inf'), []), (float('inf'), [])
    for cur_iter in range(L):
        s_inds = [(cur_iter + 2*idx)%L for idx in range(k)]
        local_best, local_best_invalid = __subroutine(0)
        if local_best[0] < best[0]:
            best = local_best
        if local_best_invalid[0] < best_invalid[0]:
            best_invalid = local_best_invalid

    return best, best_invalid
    

def apply_splitting(
    IG:IsoGraph, T_list:List[IsoGraph], Iu:IsoLine, Iv:IsoLine, 
    s_list:List[StitchingTuple], pf, new_layer_idx
) -> List[IsoLine]:
    ret, rec, k = [], [], len(T_list)
    for T in T_list:
        rec.append([(Iu, In, T[Iu][In]["S"]) for In in T.neighbors(Iu) if In != Iv])
        if Iv in T.nodes:
            rec[-1] += [(Iv, In, T[Iv][In]["S"]) for In in T.neighbors(Iv) if In != Iu]

    pos = (0.75*IG.nodes[Iu]["pos"][0] + 0.25*IG.nodes[Iv]["pos"][0],
           0.75*IG.nodes[Iu]["pos"][1] + 0.25*IG.nodes[Iv]["pos"][1])
    for idx, item in enumerate(rec):
        si, sj = s_list[idx], s_list[(idx+1)%k]
        pl = IsoLine.segment(Iu, si.get(Iu)[0], sj.get(Iu)[2]) + \
             IsoLine.segment(Iv, si.get(Iv)[0], sj.get(Iv)[2])[::-1]
        I = IsoLine(pl + [pl[0]], new_layer_idx, idx)
        ret.append(I)
        T_list[idx].add_node(I, pos=pos, label=f"(m{I.layer},{I.inner_idx})", weight=len(I), origin=(Iu, Iv))
        T_list[idx].remove_nodes_from([Iu, Iv])
        for Ix, In, _S in item:
            dat = []
            for i in get_intersection(_S._inds_set[Ix], si.get(Ix)[0], sj.get(Ix)[0]):
                s: StitchingTuple = _S._dat[_S._inds_map[Ix][i]]
                if In == s.Iv:
                    Iu_p_idx = pl.index(s.Iu_p)
                    _s = StitchingTuple(I, In, s.Iu_p, Iu_p_idx, s.Iv_q, s.Iv_q_idx)
                    _s.Iv_Bq_idx, _s.Iv_Bq = s.Iv_Bq_idx, s.Iv_Bq
                    # _s.Iu_Bp_idx, _s.Iu_Bp = I.B(Iu_p_idx) if s.Iu_p in Ix.coords else I.A(Iu_p_idx)
                    _s.Iu_Bp_idx, _s.Iu_Bp = pl.index(s.Iu_Bp), s.Iu_Bp
                elif In == s.Iu:
                    Iv_q_idx = pl.index(s.Iv_q)
                    _s = StitchingTuple(I, In, s.Iv_q, Iv_q_idx, s.Iu_p, s.Iu_p_idx)
                    _s.Iu_Bp_idx, _s.Iu_Bp = pl.index(s.Iv_Bq), s.Iv_Bq
                    # _s.Iu_Bp_idx, _s.Iu_Bp = I.B(Iv_q_idx) if s.Iv_q in Ix.coords else I.A(Iv_q_idx)
                    _s.Iv_Bq_idx, _s.Iv_Bq = s.Iu_Bp_idx, s.Iu_Bp
                dat.append(_s)
            
            S = StitchingTupleSet.from_raw_data(I, In, dat)
            if dat == []:
                _1d_idx = np.argmin(cdist(I.coords, In.coords))
                ridx, cidx = np.unravel_index(_1d_idx, (len(I), len(In)))
                ridx, cidx = int(ridx), int(cidx)
                s = StitchingTuple(I, In, I.coords[ridx], ridx, In.coords[cidx], cidx)
                S.shortest_path = {
                    "s": s,
                    "conn_pq": pf.query(s.Iu_p, s.Iv_q)[0][1:-1],
                    # "conn_BpBq": pf.query(s.Iu_Bp, s.Iv_Bq)[0][1:-1]
                }
            T_list[idx].add_edge(I, In, S=S, weight=0)
    
    return ret


def intersects(inds:set, st_idx:int, ed_idx:int) -> bool:
    if st_idx <= ed_idx:
        for idx in inds:
            if st_idx + 1 < idx < ed_idx - 1:
                return True
    else:
        for idx in inds:
            if idx > st_idx + 1 or idx < ed_idx - 1:
                return True
    
    return False


def get_intersection(inds:set, st_idx:int, ed_idx:int) -> List[int]:
    ret = []
    if st_idx <= ed_idx:
        for idx in inds:
            if st_idx + 1 < idx < ed_idx - 1:
                ret.append(idx)
    else:
        for idx in inds:
            if idx > st_idx + 1 or idx < ed_idx - 1:
                ret.append(idx)
    return ret


def solution_refinement(IG:IsoGraph, T_list:List[IsoGraph], R:List[IsoLine], pf) -> List[IsoGraph]:
    print("running solution refinement")

    pl_dist = {}
    for u, v in combinations(IG.nodes, 2):
        pl_dist[(u,v)] = pl_dist[(v,u)] = u.line.distance(v.line)
    M, repetitions  = [], {}
    VT = [set(T.nodes) for T in T_list]
    v_in_T_inds = {}
    for v in IG.nodes:
        Ts = []
        for i, VTi in enumerate(VT):
            if v in VTi:
                Ts.append(i)
        v_in_T_inds[v] = set(Ts)
        if len(Ts) > 1:
            heapq.heappush(M, (-len(Ts), v))
            repetitions[v] = Ts

    Ls = evaluate_solution(T_list, R)
    print(f"current makespan={max(Ls):.3f}, costs={[round(c, 3) for c in Ls]}")
    opt = (max(Ls), [T.copy() for T in T_list])
    
    U, num_splitting_applied, visited_shift = set(), 0, set()
    if not M:
        cand = add_improving_repetition(IG, Ls, T_list, visited_shift, U)
        if cand:
            T_idx, T_heaviest_idx, shift, pair = cand
            T_list[T_idx].add_node(shift, pos=IG.nodes[shift]["pos"], label=IG.nodes[shift]["label"], weight=IG.nodes[shift]["weight"])
            T_list[T_idx].add_edge(shift, pair, S=IG[shift][pair]["S"], weight=IG[shift][pair]["weight"])
            repetitions[shift] = [T_idx, T_heaviest_idx]
            heapq.heappush(M, (-len(repetitions[shift]), shift))
            print(f"adding {shift.index} to T[{T_idx}], dup with T[{T_heaviest_idx}]")
        else:
            print(f"do not found possible shift candidate")
    
    while M:
        num_reps, u = heapq.heappop(M)
        print(f"removing dup @ {u.index}, dup count = {-num_reps}")
        
        Ts_inds = repetitions[u]
        best = (float('inf'), None, None, None, None)
        best_invalid = (float('inf'), None, None, None, None)
        for v in IG.neighbors(u):
            if v not in U and len(IG[u][v]["S"]) != 0:
                Ts = [T_list[idx] for idx in Ts_inds]
                _Ts_inds = Ts_inds.copy()
                for idx in v_in_T_inds[v]:
                    if idx not in Ts_inds:
                        newtree = T_list[idx].copy()
                        newtree.add_node(u, pos=IG.nodes[u]["pos"], label=IG.nodes[u]["label"], weight=IG.nodes[u]["weight"])
                        newtree.add_edge(u, v, S=IG[u][v]["S"], weight=IG[u][v]["weight"])
                        Ts.append(newtree)
                        _Ts_inds.append(idx)
                local_best, local_best_invalid = pairwise_isovertices_splitting(IG, Ts, u, v, pl_dist)
                if local_best[0] < best[0]:
                    best = (*local_best, Ts, _Ts_inds, v)
                if local_best_invalid[0] < best_invalid[0]:
                    best_invalid = (*local_best_invalid, Ts, _Ts_inds, v)
        
        if best[0] != float('inf'):
            _, s_list, Ts, Ts_inds, v = best
        elif best_invalid[0] != float('inf'):
            _, s_list, Ts, Ts_inds, v = best_invalid
        else:
            s_list = None

        if s_list:
            num_splitting_applied += 1
            new_isoverts_list = apply_splitting(IG, Ts, u, v, s_list, pf, -num_splitting_applied)
            for idx, T_idx in enumerate(Ts_inds):
                T_list[T_idx] = Ts[idx]
                for n in new_isoverts_list:
                    if R[T_idx] == u or R[T_idx] == v:
                        R[T_idx] = new_isoverts_list[idx]
            
            U = U.union([u, v])
            if v in repetitions:
                M.remove((-len(repetitions[v]), v))
                print(f"removing dup @ {u.index} success, with {v.index} also removed")
            else:
                print(f"removing dup @ {u.index} success")
            S = set(IG.neighbors(v)).intersection([vert for _, vert in M])
            if S:
                n = max(S, key=lambda x:len(repetitions[x]))
                M.remove((-len(repetitions[n]), n))
                heapq.heappush(M, (-len(repetitions[n]), n))
        else:
            num_Ts_left = len(Ts_inds)
            for T_idx in sorted(Ts_inds, key=lambda x: (T_list[x].degree(u), -Ls[x])):
                T = T_list[T_idx]
                if T.degree(u) == 1 and num_Ts_left > 1:
                    T.remove_node(u)
                    repetitions[u].remove(T_idx)
                    num_Ts_left -= 1

            assert num_Ts_left >= 1
            if len(Ts_inds) != num_Ts_left:
                heapq.heappush(M, (-len(repetitions[u]), u))
            else:
                U.add(u)

            print(f"removing dup @ {u.index} failed, reducing {-num_reps} dups into {num_Ts_left} dups")

        Ls = evaluate_solution(T_list, R)
        print(f"current makespan={max(Ls):.3f}, costs={[round(c, 3) for c in Ls]}\n")
        if max(Ls) < opt[0]:
            opt = (max(Ls), [T.copy() for T in T_list])
        
        # check if we can shift the nodes from the heaviest subtree to light subtree
        if not M:
            cand = add_improving_repetition(IG, Ls, T_list, visited_shift, U)
            if cand:
                T_idx, T_heaviest_idx, shift, pair = cand
                T_list[T_idx].add_node(shift, pos=IG.nodes[shift]["pos"], label=IG.nodes[shift]["label"], weight=IG.nodes[shift]["weight"])
                T_list[T_idx].add_edge(shift, pair, S=IG[shift][pair]["S"], weight=IG[shift][pair]["weight"])
                repetitions[shift] = [T_idx, T_heaviest_idx]
                heapq.heappush(M, (-len(repetitions[shift]), shift))
                print(f"adding {shift.index} to T[{T_idx}], dup with T[{T_heaviest_idx}]")
            else:
                print(f"do not found possible shift candidate")
    
    return opt[1]
        

def add_improving_repetition(IG, Ls, T_list, visited:set, U:set) -> Tuple[int, int, IsoLine, IsoLine]:
    T_inds = np.argsort(Ls)
    T_heaviest = T_list[T_inds[-1]]
    for u in T_heaviest.nodes:
        if T_heaviest.degree(u) == 1 and u.layer >= 0 and u not in U:
            for v in T_heaviest.neighbors(u):
                if v.layer >= 0 and v not in U and (T_inds[-1], u, v) not in visited:
                    for T_idx in T_inds[:-1]:
                        B = get_boundary_vert(IG, T_list[T_idx])
                        if u in B.keys():
                            visited.add((T_inds[-1], u, v))
                            return T_idx, T_inds[-1], u, B[u]

                visited.add((T_inds[-1], u, v))

    return


def get_boundary_vert(IG, T) -> dict:
    B = {}
    for u in T.nodes:
        if u.layer >= 0:
            for v in IG.neighbors(u):
                if v not in T.nodes and v.layer >= 0:
                    B[v] = u
    return B


def evaluate_solution(T_list, R) -> List[float]:
    Ls = []
    for i in range(len(R)):
        pi, _ = unified_CFS(
            G = T_list[i],
            r = R[i],
            pr_idx = 1,
            selector_type = "MCS")
        Ls.append(path_length(pi))
    return Ls
