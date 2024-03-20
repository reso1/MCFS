from typing import List, Dict, Tuple
from itertools import combinations, product

import os, yaml, pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point

from MIP_MCPP.instance import Instance
from MIP_MCPP.model import Model

from MCFS.planning import Pathfinder
from MCFS.utils import path_length

from baselines.common import *


GRID_SIZE = 0.5


def checkerboard_partition(rect_dict:dict, debug=False):
    # read rectilinearized polygon
    polygon, Xs, Ys = read_polygon(rect_dict)

    # exterior, interiors = [], [[]]
    # for idx in rect_dict["exterior"]:
    #     if idx > 0:
    #         exterior.append(rect_dict["convex"][idx-1])
    #     else:
    #         exterior.append(rect_dict["concave"][-idx-1])
    # for interior_list in rect_dict["interiors"]:
    #     interior = []
    #     for idx in interior_list:
    #         if idx > 0:
    #             interior.append(rect_dict["convex"][idx-1])
    #         else:
    #             interior.append(rect_dict["concave"][-idx-1])
    #     interiors.append(interior) 

    # Xs = set([x for x, _ in rect_dict["convex"]]).union([x for x, _ in rect_dict["concave"]])
    # Ys = set([y for _, y in rect_dict["convex"]]).union([y for _, y in rect_dict["concave"]])
    # Xs, Ys = list(sorted(Xs)), list(sorted(Ys))
    # polygon = Polygon(exterior, interiors)
    
    # decompose
    R:List[Rectangle] = []
    for i in range(len(Xs)-1):
        xc, xn = Xs[i], Xs[i+1]
        for j in range(len(Ys)-1):
            yc, yn = Ys[j], Ys[j+1]
            rect = Rectangle((xc, yc), (xn, yn))
            rect_plg = Polygon(shell=[rect.lowerleft, rect.lowerright, rect.upperright, rect.upperleft])
            if polygon.contains_properly(Point(rect.lowerleft)) or \
               polygon.contains_properly(Point(rect.lowerright)) or \
               polygon.contains_properly(Point(rect.upperleft)) or \
               polygon.contains_properly(Point(rect.upperright)) or \
               polygon.intersection(rect_plg).area / rect_plg.area > 0.9:
                R.append(rect)

    for ra, rb in combinations(R, 2):
        if ra.lowerleft[0] == rb.lowerleft[0]:
            if ra.lowerleft[1] + ra.height == rb.lowerleft[1]:
                ra.top_rect, rb.bot_rect = rb, ra
            if ra.lowerleft[1] == rb.lowerleft[1] + rb.height:
                ra.bot_rect, rb.top_rect = rb, ra
        if ra.lowerleft[1] == rb.lowerleft[1]:
            if ra.lowerleft[0] + ra.width == rb.lowerleft[0]:
                ra.right_rect, rb.left_rect = rb, ra
            if ra.lowerleft[0] == rb.lowerleft[0] + rb.width:
                ra.left_rect, rb.right_rect = rb, ra

    if debug:
        ax = plt.gca()
        for r in R:
            r.draw(ax)

        plt.plot(polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1], '-k', lw=3)
        for interior in polygon.interiors:
            plt.plot(interior.coords.xy[0], interior.coords.xy[1], '-k', lw=3)

    return R


def orient_rectangles(R:List[Rectangle], init_dir:int, init_bias:int):
    o = {r:init_dir for r in R}
    bias = init_bias
    while True:
        unchecked = set(R)
        improved = False
        while unchecked:
            r = unchecked.pop()
            L = r.local_optimal_orientation
            if L == ARBITRARY and o[r] != bias:
                o[r] = 1 - o[r]
                for r_neighrbor in r.neighbor_rects:
                    unchecked.add(r_neighrbor)
            elif L != ARBITRARY and o[r] != L:
                o[r] = 1 - o[r]
                improved = True
                for r_neighrbor in r.neighbor_rects:
                    unchecked.add(r_neighrbor)
        
        if improved:
            bias = 1 - bias
        else:
            return o


def merge_rectangles(o:Dict[Rectangle, int]) -> List[Rectangle]:
    ret = []
    R = list(o.keys())
    visited = set()

    while R:
        r = R.pop()
        if r in visited:
            continue
        
        visited.add(r)
        
        if o[r] == VERTICAL:
            min_y, max_y = r.lowerleft[1], r.upperright[1]
            stack = [r.top_rect, r.bot_rect]
        else:
            min_x, max_x = r.lowerleft[0], r.upperright[0]
            stack = [r.left_rect, r.right_rect]

        while stack:
            n = stack.pop()
            if n and n not in visited:
                if o[r] == VERTICAL:
                    if o[n] == VERTICAL:
                        visited.add(n)
                        min_y = min(n.lowerleft[1], min_y)
                        max_y = max(n.upperright[1], max_y)
                        stack.extend([n.top_rect, n.bot_rect])
                else:
                    if o[n] != VERTICAL:
                        visited.add(n)
                        min_x = min(n.lowerleft[0], min_x)
                        max_x = max(n.upperright[0], max_x)
                        stack.extend([n.left_rect, n.right_rect])

        if o[r] == VERTICAL:
            merged = Rectangle((r.lowerleft[0], min_y), (r.upperright[0], max_y))
        else:
            merged = Rectangle((min_x, r.lowerleft[1]), (max_x, r.upperright[1]))

        ret.append(merged)

    return ret


def rank_partition(rect_dict:dict) -> List[Rectangle]:
    P = checkerboard_partition(rect_dict)
    opt = (len(P), P)
    dir = [HORIZONTAL, VERTICAL]
    for init_dir, init_bias in product(dir, dir):
        o = orient_rectangles(P, init_dir, init_bias)
        M = merge_rectangles(o)
        if len(M) < opt[0]:
            opt = (len(M), M)

    print(f"optimal paritition has size of {opt[0]}")
    return opt[1]


def get_R(P:List[Rectangle], rect_dict:dict) -> List[int]:
    ret = {}
    R = rect_dict["R"]
    for r in R:
        min_dist_pair = (float('inf'), None)
        for i in range(len(P)):
            polygon = Polygon([P[i].lowerleft, P[i].lowerright, P[i].upperright, P[i].upperleft])
            dist = polygon.distance(Point(r))
            if dist < min_dist_pair[0]:
                min_dist_pair = (dist, i)
        ret[tuple(r)] = min_dist_pair[1]

    assert len(ret.values()) == len(R)
    return list(ret.values())


def plan(instance_name:str) -> list:
    with open(os.path.join("data", "instances", instance_name+".yaml")) as f:
        dict = yaml.load(f, yaml.Loader)
        interval = float(dict["interval"])
        with open(f"data/polygons/{dict['polygon']}", "rb") as f:
            polygon = pickle.load(f)

    with open(f"data/rectilinearified/{instance_name}.rect") as f:
        dict = yaml.load(f, yaml.Loader)

    P = rank_partition(dict)
    R = get_R(P, dict)
    pf = build_pathfinder(polygon, interval)

    Pi = []
    for i, pi in enumerate(solve(P, R, pf)):
        p = pf.find_nearest(pi[-1])
        q = pf.find_nearest(dict["R"][i])
        s = pf.find_nearest(dict["R"][i])
        t = pf.find_nearest(pi[0])
        pi = pf.query(s, t)[0] + pi + pf.query(p, q)[0]
        Pi.append(np.array(pi))

    return Pi


def solve(P:List[Rectangle], R:List[int], pf:Pathfinder, interval=0.1) -> list:
    G = nx.Graph()
    for i, j in combinations(range(len(P)), 2):
        ra, rb = P[i], P[j]
        pi_a = ra.boustrophedon_path(interval)
        pi_b = rb.boustrophedon_path(interval)
        min_dist_pair = (float('inf'), None, None)
        for s, t in product([pi_a[0], pi_a[-1]], [pi_b[0], pi_b[-1]]):
            dist = np.hypot(s[0]-t[0], s[1]-t[1])
            if dist < min_dist_pair[0]:
                min_dist_pair = (dist, s, t)

        G.add_node(i, weight=path_length(pi_a))
        G.add_node(j, weight=path_length(pi_b))
        G.add_edge(i, j, weight=min_dist_pair[0], conn={i:s, j:t})

    print(f"solving mTSP, R={R}")

    istc = Instance(G, R, "0x0-test")
    model = GurobiModel(istc)
    model.wrapup(
        args = {
            "Threads": 8,
            "TimeLimit": 1800,
            "OptimalityTol": 1e-3,
            "SoftMemLimit": 16
        }
    )
    sol_edges, sol_verts = model.solve()

    Pi = [P[r].boustrophedon_path(interval) for r in R]
    for i in istc.I:
        T = nx.Graph()
        for v in sol_verts[i]:
            T.add_node(v, weight=G.nodes[v]["weight"])
        for u, v in sol_edges[i]:
            T.add_edge(u, v, weight=G[u][v]["weight"], conn=G[u][v]["conn"])
        
        for u, v in nx.dfs_edges(T, R[i]):
            pi = P[v].boustrophedon_path(interval)
            p = pf.find_nearest(Pi[i][-1])
            q = pf.find_nearest(pi[0])
            conn, _ = pf.query(p, q)

            Pi[i].extend(conn + pi)

    return Pi


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
