from __future__ import annotations
from typing import Dict,Tuple, List
from collections import defaultdict
from itertools import combinations, product

import networkx as nx

from MCFS.isolines import IsoLine, LayeredIsoLines
from MCFS.stitching_tuple import StitchingTupleSet


class IsoGraph(nx.Graph):
    NODE_ATTR = ["pos", "label", "weight"]
    EDGE_ATTR = ["S", "weight"]

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    @property
    def attrs(self) -> Tuple[dict, dict]:
        return {n:nx.get_node_attributes(self, n) for n in self.NODE_ATTR}, \
               {n:nx.get_edge_attributes(self, n) for n in self.EDGE_ATTR}

    @staticmethod
    def build(layered_isolines:LayeredIsoLines, bidirection=True) -> IsoGraph:
        ig = IsoGraph()

        O = defaultdict(set)
        for layer in range(layered_isolines.n_layers):
            for Iu in layered_isolines.at(layer):
                for idx, p in enumerate(Iu.coords):
                    ig.cross_layer_nearest_pts(p, idx, Iu, layered_isolines.at(layer+1), O) # layer => layer+1
                    ig.cross_layer_nearest_pts(p, idx, Iu, layered_isolines.at(layer-1), O) # layer => layer-1
        ig.__build_edges(O, bidirection)

        for v in ig.nodes:
            ig.nodes[v]["pos"] = (v.inner_idx, v.layer + (0.125 if v.inner_idx%2 else -0.125))
            ig.nodes[v]["label"] = (v.layer, v.inner_idx)
            ig.nodes[v]["weight"] = len(v)
            
        return ig

    @staticmethod
    def cross_layer_nearest_pts(p:tuple, p_idx:int, Iu:IsoLine, isolines:List[IsoLine], O:dict) -> None:
        p_nearest = (float("inf"), None, None, None)
        for Iv in isolines:
            dist, q_idx, q = Iv.nearest(p)
            if dist < p_nearest[0]:
                p_nearest = (dist, Iv, q, q_idx)
        if p_nearest[0] != float("inf"):
            if Iu.layer < p_nearest[1].layer:
                s = (p, p_idx, p_nearest[2], p_nearest[3])
            elif Iu.layer > p_nearest[1].layer:
                s = (p_nearest[2], p_nearest[3], p, p_idx)
            O[(Iu, p_nearest[1])].add(s)

    def __build_edges(self, O, bidirection) -> Dict[Tuple[IsoLine, IsoLine], set]:
        visited = set()
        if bidirection:
            for key, val in O.items():
                Iu, Iv = key
                if (Iu, Iv) not in visited and (Iv, Iu) in O:
                    S = StitchingTupleSet.build(Iu, Iv, val, O[(Iv, Iu)])
                    visited = visited.union([(Iu, Iv), (Iv, Iu)])
                    if len(S) != 0:
                        self.add_edge(Iu, Iv, S=S, weight=0)
        else:
            for key, val in O.items():
                Iu, Iv = key
                if (Iu, Iv) not in visited or (Iv, Iu) not in visited:
                    S = StitchingTupleSet.build(Iu, Iv, val, O[(Iu, Iv)])
                    visited = visited.union([(Iu, Iv), (Iv, Iu)])
                    if len(S) != 0:
                        self.add_edge(Iu, Iv, S=S, weight=0)
    
    def draw(self, ax, node_color="c", node_label=False, edge_label=False, R=[]) -> None:
        pos = nx.get_node_attributes(self, 'pos')
        nx.draw(self, pos, ax=ax, with_labels=False, node_color=node_color, alpha=0.8)  
        
        if node_label:
            n_labels = nx.get_node_attributes(self, 'label')
            nx.draw_networkx_labels(self, pos, ax=ax, labels=n_labels, font_size=8)

        if edge_label:
            e_labels = nx.get_node_attributes(self, 'weight')
            nx.draw_networkx_edge_labels(self, pos, ax=ax, edge_labels=e_labels, font_size=8)
        
        if type(R) is not list:
            R = [R]

        for r in R:
            ax.plot(pos[r][0], pos[r][1], f'*{node_color}', ms=32)

    def draw_isolines(self, ax) -> None:
        for v in self.nodes:
            ax.plot(v.coords.xy[0], v.coords.xy[1], '-k')
        
        for u, v in self.edges:
            S: StitchingTupleSet = self[u][v]["S"]
            for s in S.iterate():
                ax.plot([s.Iu_p[0], s.Iv_q[0]], [s.Iu_p[1], s.Iv_q[1]], '.--r')
