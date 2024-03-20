from __future__ import annotations
import os, yaml, pickle
from typing import List, Tuple, Set
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point

from MSTC_Star.utils.disjoint_set import DisjointSet
from MSTC_Star.mcpp.mstc_star_planner import MSTCStarPlanner

from baselines.common import *


GRID_SIZE = 0.2
KEY = lambda ra, rb: (ra, rb) if ra.lowerleft < rb.lowerleft else (rb, ra)
CVT = lambda p: (FIX(int(p[0] / GRID_SIZE) * GRID_SIZE), 
                 FIX(int(p[1] / GRID_SIZE) * GRID_SIZE))


def get_graphs(rect_dict:dict) -> Tuple[nx.Graph, nx.Graph]:
    # read rectilinearized polygon
    polygon, _, _ = read_polygon(rect_dict)

    x_min, y_min, x_max, y_max = polygon.bounds
    x_min, y_min, x_max, y_max = FIX(x_min), FIX(y_min), FIX(x_max), FIX(y_max)

    # decompose
    R:List[Rectangle] = []
    for xc in np.arange(x_min, x_max, GRID_SIZE):
        xn = xc + GRID_SIZE
        for yc in np.arange(y_min, y_max, GRID_SIZE):
            yn = yc + GRID_SIZE
            rect = Rectangle((xc, yc), (xn, yn))
            rect_plg = Polygon(shell=[rect.lowerleft, rect.lowerright, rect.upperright, rect.upperleft])
            if polygon.contains_properly(Point(rect.lowerleft)) or \
               polygon.contains_properly(Point(rect.lowerright)) or \
               polygon.contains_properly(Point(rect.upperleft)) or \
               polygon.contains_properly(Point(rect.upperright)) or \
               polygon.intersection(rect_plg).area / rect_plg.area > 0.9:
                R.append(rect)

    I, E, V, H = nx.DiGraph(), set(), set(), set()
    for ra, rb in combinations(R, 2):
        if ra.lowerleft[0] == rb.lowerleft[0]:
            if FIX(ra.lowerleft[1] + ra.height) == rb.lowerleft[1]:
                key = KEY(ra, rb)
                E.add(key)
                v = (ra.lowerleft, rb.lowerleft)
                V.add(v)
                I.add_node(v, pair=key, bipartite=VERTICAL)
            if ra.lowerleft[1] == FIX(rb.lowerleft[1] + rb.height):
                key = KEY(ra, rb)
                E.add(key)
                v = (ra.lowerleft, rb.lowerleft)
                V.add(v)
                I.add_node(v, pair=key, bipartite=VERTICAL)
           
        if ra.lowerleft[1] == rb.lowerleft[1]:
            if FIX(ra.lowerleft[0] + ra.width) == rb.lowerleft[0]:
                key = KEY(ra, rb)
                E.add(key)
                h = (ra.lowerleft, rb.lowerleft)
                H.add(h)
                I.add_node(h, pair=key, bipartite=HORIZONTAL)
            if ra.lowerleft[0] == FIX(rb.lowerleft[0] + rb.width):
                key = KEY(ra, rb)
                E.add(key)
                h = (ra.lowerleft, rb.lowerleft)
                H.add(h)
                I.add_node(h, pair=key, bipartite=HORIZONTAL)

    I.graph["V"], I.graph["H"] = V, H
    for h, v in product(H, V):
        if v[0] == h[0] or v[0] == h[1] or v[1] == h[0] or v[1] == h[1]:
            I.add_edge(h, v, capacity=1)
    
    return I, nx.Graph(E)


def bipartite_min_vertex_cover(I:nx.Graph, G:nx.Graph) -> List[Rectangle]:
    # find maximum independent set
    src, dst = "s", "t"
    I.add_edges_from([(src, h) for h in I.graph["H"]], capacity=1)
    I.add_edges_from([(v, dst) for v in I.graph["V"]], capacity=1)

    residual, H, V = nx.Graph(), set(), set()
    _, flow_dict = nx.maximum_flow(I, src, dst)
    for u, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 0:
                residual.add_edge(u, v)
                if u != src and u != dst:
                    if I.nodes[u]["bipartite"] == HORIZONTAL:
                        H.add(u)
                    else:
                        V.add(u)
                if v != src and v != dst:
                    if I.nodes[v]["bipartite"] == HORIZONTAL:
                        H.add(v)
                    else:
                        V.add(v)
    
    C, D = set(), set()
    for h in H:
        if not nx.has_path(residual, src, h):
            C.add(h)
    for v in V:
        if nx.has_path(residual, src, v):
            D.add(v)
    
    C, D = set(I.graph["H"]) - C, set(I.graph["V"]) - D 
    
    # merge
    ds, ds_nodes = DisjointSet(), {}
    for grid in G.nodes:
        ds_nodes[grid] = ds.make(grid)

    for v in set.union(C, D):
        grid_a, grid_b = I.nodes[v]["pair"]
        ds.union(ds_nodes[grid_a], ds_nodes[grid_b])

    roots = defaultdict(list)
    hash = lambda node: node.data.__hash__()
    for grid in G.nodes:
        roots[hash(ds.find(ds_nodes[grid]))].append(grid)

    B:List[Rectangle] = []
    for _, val in roots.items():
        val.sort(key=lambda x:x.lowerleft[0])
        val.sort(key=lambda x:x.lowerleft[1])
        Rectangle: Rectangle = val[0].copy()
        for grid in val[1:]:
            if Rectangle.merge(grid):
                Rectangle = Rectangle.merge(grid)
            else:
                B.append(Rectangle)
                Rectangle = grid
        B.append(Rectangle)

    return B


def merge_Rectangles(B:List[Rectangle]) -> nx.Graph:
    grids, T = {}, nx.Graph()
    for Rectangle in B:
        assert Rectangle.width == GRID_SIZE or Rectangle.height == GRID_SIZE
        grids[Rectangle] = Rectangle.grids(GRID_SIZE)
        if len(grids[Rectangle]) == 1:
            T.add_node(grids[Rectangle][0])

        for grid_a, grid_b in combinations(grids[Rectangle], 2):
            if grid_a.is_adjacent(grid_b):
                T.add_edge(grid_a, grid_b)

    E = defaultdict(set)
    for Rectangle_a, Rectangle_b in combinations(B, 2):
        for grid_a, grid_b in product(grids[Rectangle_a], grids[Rectangle_b]):
            if grid_a.is_adjacent(grid_b):
                E[KEY(Rectangle_a, Rectangle_b)].add(KEY(grid_a, grid_b))

    def f(grid:Rectangle) -> int:
        deg = T.degree(grid)
        assert 0 <= deg <= 4
        if deg == 1 or deg == 3:
            return 2
        if deg == 0 or deg == 4:
            return 4
        if deg == 2:
            return 0
            # ngb_a, ngb_b = list(T.neighbors(grid))
            # if grid.left_rect == ngb_a and grid.right_rect == ngb_b or \
            #    grid.right_rect == ngb_a and grid.left_rect == ngb_b or \
            #    grid.top_rect == ngb_a and grid.bot_rect == ngb_b or \
            #    grid.bot_rect == ngb_a and grid.top_rect == ngb_b:
            #     return 0
            # else:
            #     return 2
    
    def g(grid_a:Rectangle, grid_b:Rectangle) -> int:
        ret = - f(grid_a) - f(grid_b)
        T.add_edge(grid_a, grid_b)
        ret += f(grid_a) + f(grid_b)
        T.remove_edge(grid_a, grid_b)
        return ret

    S = set()
    while not nx.is_connected(T):
        min_g_pair = (float('inf'), None, None)
        for key, val in E.items():
            sub_E = list(val)
            gs = [g(grid_a, grid_b) for grid_a, grid_b in sub_E]
            idx = np.argmin(gs)
            if gs[idx] < min_g_pair[0]:
                min_g_pair = (gs[idx], key, sub_E[idx])
        
        if not nx.has_path(T, min_g_pair[2][0], min_g_pair[2][1]):
            T.add_edge(*min_g_pair[2])
            S = S.union(min_g_pair[1])

        E.pop(min_g_pair[1])

    return T


def get_R(T:nx.Graph, rect_dict:dict) -> List[int]:
    ret = {}
    R = rect_dict["R"]
    for r in R:
        min_dist_pair = (float('inf'), None)
        for v in T.nodes:
            polygon = Polygon([v.lowerleft, v.lowerright, v.upperright, v.upperleft])
            dist = polygon.distance(Point(r))
            if dist < min_dist_pair[0]:
                min_dist_pair = (dist, v)
        ret[tuple(r)] = min_dist_pair[1]

    assert len(ret.values()) == len(R)
    return ret


def plan(instance_name:str) -> list:
    with open(os.path.join("data", "instances", instance_name+".yaml")) as f:
        dict = yaml.load(f, yaml.Loader)
        interval = float(dict["interval"])
        with open(f"data/polygons/{dict['polygon']}", "rb") as f:
            polygon = pickle.load(f)

    with open(f"data/rectilinearified/{instance_name}.rect") as f:
        dict = yaml.load(f, yaml.Loader)

    B = bipartite_min_vertex_cover(*get_graphs(dict))
    T = merge_Rectangles(B)
    
    R = get_R(T, dict)
    pf = build_pathfinder(polygon, interval)

    G, nodes = nx.Graph(), {}
    for u, v in T.edges:
        gu = (u.lowerleft[0]/GRID_SIZE, u.lowerleft[1]/GRID_SIZE)
        gv = (v.lowerleft[0]/GRID_SIZE, v.lowerleft[1]/GRID_SIZE)
        nodes[u], nodes[v] = gu, gv
        G.add_edge(gu, gv, weight=1)
        G.nodes[gu]["terrain_weight"] = G.nodes[gv]["terrain_weight"] = 1

    planner = MSTCStarPlanner(G, len(R), [nodes[r] for r in list(R.values())], float('inf'))
    plans = planner.allocate()

    Pi = []
    for i, pi in enumerate(plans.values()):
        pi = np.array(pi) * GRID_SIZE + np.array([GRID_SIZE/2, GRID_SIZE/2])
        s = pf.find_nearest(dict["R"][i])
        t = pf.find_nearest(pi[0])
        p = pf.find_nearest(pi[-1])
        q = pf.find_nearest(dict["R"][i])
        Pi.append(np.vstack([pf.query(s, t)[0], pi, pf.query(p, q)[0]]))

    return Pi
