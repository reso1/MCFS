from __future__ import annotations
from typing import List, Tuple
from collections import defaultdict
import heapq
import math

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from shapely.geometry import LineString

from MCFS.utils import segment
from MCFS.isolines import IsoLine, LayeredIsoLines
from MCFS.isograph import IsoGraph
from MCFS.selector import Selector
from MCFS.stitching_tuple import StitchingTuple


class Pathfinder:
    def __init__(self, IG:IsoGraph, interval:float) -> None:
        self.G = nx.Graph()
        for u in IG.nodes:
            self.G.add_edges_from([(u.coords[i],u.coords[i+1])for i in range(len(u)-1)])
        for u, v in IG.edges:
            self.G.add_edges_from([(s.Iu_p, s.Iv_q) for s in IG[u][v]["S"].iterate()])
        self.heur = lambda p,q: abs(p[0]-q[0]) + abs(p[1]-q[1])
        self.interval = interval
    
    def query(self, p:tuple, q:tuple) -> Tuple[list, float]:
        pi = nx.astar_path(self.G, p, q, self.heur)
        return pi, self.interval * (len(pi)-1)
    
    def find_nearest(self, pts:tuple) -> tuple:
        N = list(self.G.nodes)
        D = [math.hypot(p[0]-pts[0], p[1]-pts[1]) for p in N]
        return N[np.argmin(D)]


def unified_CFS(
    G:IsoGraph, r:IsoLine, pr_idx:int, selector_type:str
) -> Tuple[list, List[StitchingTuple]]:
    
    pi = r.segment(pr_idx)
    U = set([(pi[0], pi[-1])])
    stitching_tuples = []

    # fig, ax = plt.subplots(1,2)
    # plt.ion()
    # plt.show()
    # sub_pi = np.array(pi)
    # ax[0].plot(sub_pi[:, 0], sub_pi[:, 1], f'-', color='k')
    # ax[0].plot(sub_pi[:, 0], sub_pi[:, 1], f'.', color='gray', alpha=0.6)
    # ax[0].plot(sub_pi[0, 0], sub_pi[0, 1], '.r')
    # ax[0].plot(sub_pi[-1, 0], sub_pi[-1, 1], '.b')

    # ax[1].plot(sub_pi[0, 0], sub_pi[0, 1], '^r')
    # ax[1].plot(sub_pi[-1, 0], sub_pi[-1, 1], 'sb', mfc='none')
    # ax[1].plot(sub_pi[:, 0], sub_pi[:, 1], '-k')

    selector = Selector(pi[0], r)

    for Iu, Iv in nx.dfs_edges(G, r):
        if len(G[Iu][Iv]["S"]) == 0:
            pi_dict = G[Iu][Iv]["S"].shortest_path
            s = pi_dict["s"]
            conn_pq = pi_dict["conn_pq"]
            # conn_pq, conn_BpBq = pi_dict["conn_pq"], pi_dict["conn_BpBq"]
            if Iu != G[Iu][Iv]["S"].Iu:
                conn_pq = conn_pq[::-1]
                # conn_pq, conn_BpBq = conn_pq[::-1], conn_BpBq[::-1]
        else:
            conn_pq, conn_BpBq = [], []
            s = selector.select(selector_type, G, Iu, Iv, U)

        _, Iu_p, _, Iu_Bp = s.get(Iu)
        Iv_q_idx, Iv_q, Iv_Bq_idx, Iv_Bq = s.get(Iv)

        Iu_p_idx = pi.index(Iu_p)
        pi_v = Iv.segment(Iv_q_idx if Iv.B(Iv_q_idx)[0] == Iv_Bq_idx else Iv_Bq_idx)
        pi_v = conn_pq + pi_v + conn_pq[::-1]

        if pi[(Iu_p_idx-1)%len(pi)] == Iu_Bp:
            if Iv.B(Iv_q_idx)[0] == Iv_Bq_idx:
                pi = pi[:Iu_p_idx] + pi_v[::-1] + pi[Iu_p_idx:]
            else:
                pi = pi[:Iu_p_idx] + pi_v + pi[Iu_p_idx:]
        else:
            if Iv.B(Iv_q_idx)[0] == Iv_Bq_idx:
                pi = pi[:Iu_p_idx+1] + pi_v + pi[Iu_p_idx+1:]
            else:
                pi = pi[:Iu_p_idx+1] + pi_v[::-1] + pi[Iu_p_idx+1:]
        
        # pi_v = np.array(pi_v[::-1] if pi[(Iu_p_idx-1)%len(pi)] == Iu_Bp else pi_v)
        # ax[0].plot(pi_v[0, 0], pi_v[0, 1], f'^r', mfc='none')
        # ax[0].plot(pi_v[-1, 0], pi_v[-1, 1], f'sb', mfc='none')
        # ax[0].plot(pi_v[:, 0], pi_v[:, 1], f'-', color='k')
        # ax[0].plot(pi_v[:, 0], pi_v[:, 1], f'.', color='gray', alpha=0.8)
        # ax[0].plot([Iu_p[0], Iv_q[0]], [Iu_p[1], Iv_q[1]], 'xr')
        # ax[0].plot([Iu_Bp[0], Iv_Bq[0]], [Iu_Bp[1], Iv_Bq[1]], 'xb')
        # ax[0].set_aspect('equal', adjustable='box')
        # ax[0].grid(True)
        
        # _pi = np.array(pi)
        # ax[1].cla()
        # ax[1].plot(_pi[0, 0], _pi[0, 1], '^r', mfc='none')
        # ax[1].plot(_pi[-1, 0], _pi[-1, 1], 'sb', mfc='none')
        # ax[1].plot(_pi[:, 0], _pi[:, 1], f'-k')
        # ax[1].set_aspect('equal', adjustable='box')
        # ax[1].grid(True)
        # ax[1].plot([Iu_p[0], Iv_q[0]], [Iu_p[1], Iv_q[1]], '.k')
        # ax[1].plot([Iu_Bp[0], Iv_Bq[0]], [Iu_Bp[1], Iv_Bq[1]], '.k')

        U = U.union([(Iu_p, Iu_Bp), (Iv_q, Iv_Bq)])
        stitching_tuples.append(s)
    
    if selector_type == "CFS":
        pi = pi[1:] + pi[:1]
    
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal', adjustable='box')
    # _pi = np.array(pi)
    # ax.plot(_pi[:, 0], _pi[:, 1], f'-k')
    # ax.plot(_pi[0, 0], _pi[0, 1], f'^k')
    # ax.plot(_pi[-1, 0], _pi[-1, 1], f'^k')
    # for Iu_p, Iv_q, Iu_Bp, Iv_Bq in stitching_tuples:
    #     ax.plot([Iu_p[0], Iv_q[0]], [Iu_p[1], Iv_q[1]], 'ok', ms=4)
    #     ax.plot([Iu_Bp[0], Iv_Bq[0]], [Iu_Bp[1], Iv_Bq[1]], 'ok', ms=4)
    #     ax.plot([Iu_p[0], Iv_q[0]], [Iu_p[1], Iv_q[1]], '-b', lw=1.5)
    #     ax.plot([Iu_Bp[0], Iv_Bq[0]], [Iu_Bp[1], Iv_Bq[1]], '-b', lw=1.5)
    #     ax.plot([Iu_p[0], Iu_Bp[0]], [Iu_p[1], Iu_Bp[1]], '--r', lw=1.5)
    #     ax.plot([Iv_q[0], Iv_Bq[0]], [Iv_q[1], Iv_Bq[1]], '--r', lw=1.5)

    return pi, stitching_tuples

