# MCFS
This repository is the implementation of the unified version of Connected Fermat Spiral and the Multi-Robot Connected Fermat Spiral with its two optimizations from the following paper:

*Jingtao Tang and Hang Ma. "Multi-Robot Connected Fermat Spiral Coverage." ICAPS 2024 (to appear). [[paper]](https://arxiv.org/abs/2403.13311), [[simulation]](), [[project]]()*

Please cite this paper if you use this code for the multi-robot coverage path planning problem.

## Installation
`pip install -r requirements.txt`

## Usage
```bash
python main.py [-h] [--aug AUG] [--ref REF] [--read_mmrtc_sol READ_MMRTC_SOL] [--write_sol WRITE_SOL] [--draw DRAW] istc
```
- Required:
  - `istc`: the instance name stored in directory 'data/instances'.
- Optional:
  - `--aug AUG`: Is using optimization of isograph augmentation (default=True)
  - `--ref REF`: Is using optimization of MMRTC solution refinement (default=True)
  - `--read_mmrtc_sol READ_MMRTC_SOL`: Is reading prerun MMRTC solution (default=True)
  - `--write_sol WRITE_SOL`: Is writing the MMRTC and MCPP solutions (default=False)
  - `--draw DRAW`: Is drawing the MCPP solution (default=False)

## File Structure
- main.py: main function
- exp_ablation_study.ipynb: code for ablation study
- exp_case_study.ipynb: code for case study
- exp_comparison.ipynb: code for performance evaluation with two other baseline methods
- MIP-MCPP: repo of the work "*Mixed Integer Programming for Time-Optimal Multi-Robot Coverage Path Planning With Efficient Heuristics*"
- MCFS/
  - isograph.py: the isograph class representing the set of contouring isolines
  - isolines.py: two classes of the individual isoline and set of layered contouring isolines
  - MMRTC.py: wrappers of Gurobi and SCIP solvers for MMRTC models
  - planning.py: the unified version of CFS and a simple A* pathfinder
  - selector.py: the class of stitching tuple selectors for the unified version of CFS
  - solution_refinement.py: the second optimization of MMRTC solution refinement, including the main algorithm and the two subroutines of add_improving_repetition and pairwise_isovertices_splitting
  -  stitching_tuple.py: the class of individual stitching tuple and the set $O$ of stitching tuples 
- data/
  - instances: metadata of the MCPP instances
  - polygons: the respective [shapely polygon](https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html) objects of the MCPP instances (for MCFS)
  - rectilinearified: the rectilinearized grid approximations of the MCPP instances (for the two comparing baselines) 
  - solutions: MMRTC and MCPP solutions for MCFS w/ two optimizations of isograph augmentation and solution refinement

## License
MCFS is released under the GPL version 3. See LICENSE.txt for further details.
