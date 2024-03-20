from typing import List
from collections import defaultdict

import networkx as nx

import pyscipopt
from pyscipopt import quicksum

from MIP_MCPP.instance import Instance
from MIP_MCPP.model import Model
from MIP_MCPP.warmstarter import WarmStarter
from MIP_MCPP.misc import uv_sorted
from MIP_MCPP.warmstarter import gen_flow

from MCFS.isograph import IsoGraph
from MCFS.stitching_tuple import StitchingTuple, StitchingTupleSet


def augmented_isograph(IG:IsoGraph, interval:float, delta:int=2) -> IsoGraph:
    assert delta >= 2
    # get all k_hop pairs of isoverts
    k_hop_pairs = set()
    shortest_paths = nx.shortest_path(IG)
    for src, val in shortest_paths.items():
        for dst, path in val.items():
            if len(path) - 1 == delta and (src, dst) not in k_hop_pairs \
                                  and (dst, src) not in k_hop_pairs:
                k_hop_pairs.add((src, dst))

    # create new edges
    def __subroutine(
        pi:list, i:int, s_last:List[StitchingTuple], ret:dict
    ) -> None:
        if i == delta:
            return

        Iu, Iv = pi[i], pi[i+1]
        S:StitchingTupleSet = IG[Iu][Iv]["S"]
        a_idx, a, b_idx, b = s_last[-1].get(Iu)
        for s in S.iterate():
            na_idx, na, nb_idx, nb = s.get(Iu)
            if a_idx == na_idx and b_idx == nb_idx:
                ret[i].append(s_last+[s])
                __subroutine(pi, i+1, s_last+[s], ret)

    for u, v in k_hop_pairs:
        ret = defaultdict(list)
        pi = shortest_paths[u][v]
        S:StitchingTupleSet = IG[pi[0]][pi[1]]["S"]
        for s in S.iterate():
            __subroutine(pi, 1, [s], ret)

        for i, s_list in ret.items():
            dat = []
            for val in s_list:
                dat.append(StitchingTuple(
                    Iu = pi[0], Iv = pi[i+1],
                    Iu_p = val[0].get(pi[0])[1],
                    Iu_pidx = val[0].get(pi[0])[0],
                    Iv_q = val[-1].get(pi[i+1])[1],
                    Iv_qidx = val[-1].get(pi[i+1])[0]
                ))
            IG.add_edge(pi[0], pi[i+1], 
                S=StitchingTupleSet.from_raw_data(pi[0],pi[i+1],dat),
                weight=2*i*interval
            )
    
    return IG


def solve_MMRTC_model_SCIP(IG:IsoGraph, R:list) -> List[IsoGraph]: 
    istc = Instance(IG, R, "0x0-test")
    rmmtc_model = SCIPModel(istc)
    n_attrs, e_attrs = IG.attrs
    
    rmmtc_model.warmstart()

    ret = []
    for T in rmmtc_model.solve():
        for n, attr in n_attrs.items():
            nx.set_node_attributes(T, attr, n)
        for n, attr in e_attrs.items():
            nx.set_edge_attributes(T, attr, n)
        ret.append(IsoGraph(T))

    return ret


class SCIPModel:

    def __init__(self, istc:Instance) -> None:
        self.istc = istc
        self.model = pyscipopt.Model("MMRTC")
        self.v2id = {v:idx for idx, v in enumerate(self.istc.G.nodes())}
        self.id2v = {idx:v for idx, v in enumerate(self.istc.G.nodes())}
        self.w_V = list(nx.get_node_attributes(self.istc.G, "weight").values())
        self.w_E = list(nx.get_edge_attributes(self.istc.G, "weight").values())
        self._add_vars()
        self._add_constrs()
        self.model.setObjective(self.tau)
        self.model.setRealParam('limits/gap', 0.1)
        self.model.setParam('limits/time', 1800)
    
    def _add_vars(self) -> None:
        """ Variables """
        self.tau = self.model.addVar("tau", vtype="CONTINUOUS")
        self.x, self.y, self.fu, self.fv = [], [], [], []
        for i in self.istc.I:
            _x, _y, _fu, _fv = [], [], [], []
            for v in self.istc.V:
                _y.append(self.model.addVar(f"y{i,v}", vtype="BINARY"))
            for e in self.istc.E:
                _x.append(self.model.addVar(f"x{i,e}", vtype="BINARY"))
                _fu.append(self.model.addVar(f"fu{i,e}", vtype="CONTINUOUS"))
                _fv.append(self.model.addVar(f"fv{i,e}", vtype="CONTINUOUS"))
            self.x.append(_x)
            self.y.append(_y)
            self.fu.append(_fu)
            self.fv.append(_fv)

    def _add_constrs(self) -> None:
        
        for i in self.istc.I:
            # CONSTRAINTS(makespan):        sum_v{w_v * y_v^i} <= tau
            self.model.addCons( quicksum(self.x[i][e]*self.w_E[e] for e in self.istc.E) +
                                quicksum(self.y[i][v]*self.w_V[v] for v in self.istc.V) <= self.tau )
            # CONSTRAINTS(rooted):          y_ri^i = 1
            self.model.addCons( self.y[i][self.v2id[self.istc.R[i]]] == 1 )
            # CONSTRAINTS(tree):            sum_v{y_v^i} = 1 + sum_e{x_e^i}
            self.model.addCons( quicksum(self.y[i][v] for v in self.istc.V) == 
                                1 + quicksum(self.x[i][e] for e in self.istc.E) )
            # CONSTRAINTS(flow):            fu_e^i + fv_e^i = x_e^i
            for e in self.istc.E:
                self.model.addCons( self.fu[i][e] + self.fv[i][e] == self.x[i][e] )

        for v in self.istc.V:
            # CONSTRAINTS(cover):           sum_i{y_v^i} >= 1
            self.model.addCons( quicksum(self.y[i][v] for i in self.istc.I) >= 1 )
            
            for i in self.istc.I:
                sum_flow, _v = 0, self.id2v[v]
                for u in self.istc.G.neighbors(_v):
                    uv = uv_sorted((u, _v))
                    e = self.istc.uv2e[uv]
                    sum_flow += (self.fu[i][e] if _v == uv[0] else self.fv[i][e])
                    # CONSTRAINTS(y_def):   x_e^i <= y_v^i
                    self.model.addCons( self.x[i][e] <= self.y[i][v])
                
                # CONSTRAINTS(acyclic):     sum_e{fv_e^i} <= 1 - 1 / |V|
                self.model.addCons( sum_flow <= 1 - 1 / self.istc.n ) 
        
    def solve(self) -> List[nx.Graph]:
        try:
            self.model.optimize()
        except Exception as e:
            print(e)

        T = [nx.Graph() for _ in self.istc.I]
        for i in self.istc.I:
            for v in self.istc.V:
                if self.model.getVal(self.y[i][v]) > 0.5:
                    n = self.id2v[v]
                    T[i].add_node(n, weight=self.istc.G.nodes[n]["weight"])
            for e in self.istc.E:
                if self.model.getVal(self.x[i][e]) > 0.5:
                    u, v = self.istc.e2uv[e]
                    T[i].add_edge(u, v, weight=self.istc.G[u][v]["weight"])

            assert nx.is_connected(T[i])

        return T

    def warmstart(self) -> None:
        sol = self.model.createSol()
        M = nx.minimum_spanning_tree(self.istc.G)
        f_eu, f_ev = gen_flow(M.copy())        
        tau = 0
        for u, v in M.edges():
            uv = uv_sorted((u, v))
            eid = self.istc.uv2e[uv]
            uid, vid = self.v2id[u], self.v2id[v]
            tau += self.w_E[eid] + self.w_V[uid] + self.w_V[vid]
            for i in self.istc.I:
                self.model.setSolVal(sol, self.x[i][eid], 1)
                self.model.setSolVal(sol, self.y[i][uid], 1)
                self.model.setSolVal(sol, self.y[i][vid], 1)
                self.model.setSolVal(sol, self.fu[i][eid], f_eu[uv])
                self.model.setSolVal(sol, self.fv[i][eid], f_ev[uv])

        self.model.setSolVal(sol, self.tau, tau)
        self.model.addSol(sol)


def solve_MMRTC_model_GRB(IG:IsoGraph, R:list, ) -> List[IsoGraph]:

    IG_idx = nx.Graph()
    iv2idx = {iv:idx for idx, iv in enumerate(IG.nodes)}
    idx2iv = {idx:iv for idx, iv in enumerate(IG.nodes)}
    R_idx = [iv2idx[r] for r in R]
    for u, v in IG.edges:
        IG_idx.add_node(iv2idx[u], weight=IG.nodes[u]["weight"])
        IG_idx.add_node(iv2idx[v], weight=IG.nodes[v]["weight"])
        IG_idx.add_edge(iv2idx[u], iv2idx[v], weight=IG[u][v]["weight"])
        
    istc = Instance(IG_idx, R_idx, "0x0-test")
    rmmtc_model = GurobiModel(istc)
    rmmtc_model.wrapup(
        args = {
            "Threads": 8,
            "TimeLimit": 1800,
            "OptimalityTol": 1e-3,
            "SoftMemLimit": 16
        }
    )

    rmmtc_model = WarmStarter.apply(rmmtc_model, "MST", None)
    sol_edges, sol_verts = rmmtc_model.solve()

    ret = []
    n_attrs, e_attrs = IG.attrs
    for i in istc.I:
        T = nx.Graph()
        for v in sol_verts[i]:
            n = idx2iv[v]
            T.add_node(n, weight=IG.nodes[n]["weight"])
        for u, v in sol_edges[i]:
            u, v = idx2iv[u], idx2iv[v]
            T.add_edge(u, v, weight=IG[u][v]["weight"])
        for n, attr in n_attrs.items():
            nx.set_node_attributes(T, attr, n)
        for n, attr in e_attrs.items():
            nx.set_edge_attributes(T, attr, n)
        ret.append(IsoGraph(T))  

    return ret


class GurobiModel(Model):
    
    def __init__(self, istc: Instance) -> None:
        super().__init__(istc)

    def _init_constrs(self, H:List[nx.Graph]) -> None:
        super()._init_constrs(H)
        w_V = list(nx.get_node_attributes(self.istc.G, "weight").values())
        w_E = list(nx.get_edge_attributes(self.istc.G, "weight").values())
        for i in self.istc.I:
            self.C_makespan.x_coeffs[i, self.e_ind[i]:self.e_ind[i+1]] = w_E
            self.C_makespan.y_coeffs[i, self.v_ind[i]:self.v_ind[i+1]] = w_V
