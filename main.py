import os
import sys
import yaml
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MCFS

from MCFS.utils import path_length, curvature, LineDataUnits, draw_MCPP_solution
from MCFS.isograph import IsoGraph
from MCFS.MMRTC import augmented_isograph, solve_MMRTC_model_GRB
from MCFS.isolines import LayeredIsoLines
from MCFS.planning import Pathfinder, unified_CFS
from MCFS.solution_refinement import *


def MCFS(instance_name:str, aug=True, ref=True, read_mmrtc_sol=False, write_sol=False):

    with open(os.path.join("data", "instances", instance_name+".yaml")) as f:
        dict = yaml.load(f, yaml.Loader)
        delta = int(dict["delta"])
        interval = float(dict["interval"])
        R = [tuple(r) for r in dict["R"]]
        with open(f"data/polygons/{dict['polygon']}", "rb") as f:
            polygon = pickle.load(f)
    
    min_x, min_y, max_x, max_y = polygon.bounds

    t = time.time()
    layered_isolines = LayeredIsoLines.from_polygon(polygon, interval)
    IG = IsoGraph.build(layered_isolines)
    pf = Pathfinder(IG, interval)
    R = [layered_isolines.at(*index) for index in R]
    if aug:
        IG = augmented_isograph(IG, interval=interval, delta=delta)
    t2 = time.time() - t

    if aug and read_mmrtc_sol:
        with open(f"data/solutions/{instance_name}.mmrtc.sol", "rb") as f:
            T = pickle.load(f)
        t1 = -1
    else:
        t = time.time()
        # T = solve_MMRTC_model_SCIP(IG, R)
        T = solve_MMRTC_model_GRB(IG, R)
        if write_sol:
            with open(f"data/solutions/{instance_name}.mmrtc.sol", "wb") as f:
                pickle.dump(T, f)
        t1 = time.time() - t

    if ref:
        t = time.time()
        T = solution_refinement(IG, T, R, pf)
        t2 += time.time() - t

    Pi, Ls, ks = [], [], np.array([])
    for i in range(len(R)):
        pi, s_list = unified_CFS(
            G = T[i],
            r = R[i],
            pr_idx = i*10,
            selector_type = "MCS"
        )
        j = 0
        while j < len(pi) - 1:
            if pi[j] == pi[j+1]:
                pi.pop(j)
            else:
                j += 1
        # a workaround to make the path align with the boundary
        pi = np.array(pi) * np.array([1 + 0.12/(max_x-min_x), 1 + 0.12/(max_y-min_y)])
        Pi.append(pi)
        Ls.append(path_length(pi))
        ks = np.vstack(np.sqrt(curvature(pi)))
    
    if write_sol:
        with open(f"data/solutions/{instance_name}.mcpp.sol", "wb") as f:
            pickle.dump(Pi, f)
        fig, ax = draw_MCPP_solution(Pi)
        fig.savefig(f"data/solutions/{instance_name}.png", dpi=500)

    makespan, smoothness = max(Ls), np.average(ks)
    print(f"tau={makespan}, smoothness={smoothness}")
    print(f"t1:{t1:.3f}, t2:{t2:.3f}")

    return Pi, (makespan, smoothness, t1, t2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("istc", help="Instance name. Choose from {I, C, A, P, S, double_torus, office}")
    parser.add_argument("--aug", default=True, help="Is using optimization of isograph augmentation")
    parser.add_argument("--ref", default=True, help="Is using optimization of MMRTC solution refinement")
    parser.add_argument("--read_mmrtc_sol", default=True, help="Is reading prerun MMRTC solution")
    parser.add_argument("--write_sol", default=False, help="Is writing the MMRTC and MCPP solutions")
    parser.add_argument("--draw", default=False, help="Is drawing the MCPP solution")

    args = parser.parse_args()

    Pi, metrics = MCFS(
        instance_name   = args.istc,
        aug             = bool(args.aug), 
        ref             = bool(args.ref),
        read_mmrtc_sol  = bool(args.read_mmrtc_sol),
        write_sol       = bool(args.write_sol)
    )

    if bool(args.draw):
        draw_MCPP_solution(Pi)
        plt.show()
    