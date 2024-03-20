from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np
import networkx as nx

import matplotlib.patches as patch

from shapely.geometry import Polygon, Point

from MCFS.isograph import IsoGraph
from MCFS.isolines import LayeredIsoLines
from MCFS.planning import Pathfinder


HORIZONTAL = 0 
VERTICAL   = 1
ARBITRARY  = 2

FIX = lambda x: round(x, 1)


class Rectangle:

    def __init__(self, lowerleft: tuple, upperright: tuple) -> None:
        ll = (FIX(lowerleft[0]), FIX(lowerleft[1]))
        ur = (FIX(upperright[0]), FIX(upperright[1]))
        self.lowerleft = ll
        self.upperright = ur
        self.width = FIX(ur[0] - ll[0])
        self.height = FIX(ur[1] - ll[1])
        self.lowerright = (ur[0], ll[1])
        self.upperleft = (ll[0], ur[1])

        self.top_rect: Rectangle = None
        self.bot_rect: Rectangle = None
        self.left_rect: Rectangle = None
        self.right_rect: Rectangle = None

    
    def __hash__(self) -> int:
        return (*self.lowerleft, self.width, self.height).__hash__()
        
    @property
    def local_optimal_orientation(self) -> int:
        onehot = tuple([1 if x else 0 for x in [self.top_rect, self.left_rect, self.bot_rect, self.right_rect]])
        a_case, f_case = (0, 0, 0, 0), (1, 1, 1, 1)
        c_case = [tuple(np.roll([1, 1, 0, 0], i)) for i in range(4)]
        for case in [a_case, f_case] + c_case:
            if onehot == case:
                return ARBITRARY
        
        if onehot == (1, 0, 0, 0) or onehot == (0, 0, 1, 0) or onehot == (1, 0, 1, 0) or \
           onehot == (1, 1, 1, 0) or onehot == (1, 0, 1, 1):
            return VERTICAL
        
        if onehot == (0, 1, 0, 0) or onehot == (0, 0, 0, 1) or onehot == (0, 1, 0, 1) or \
           onehot == (1, 1, 0, 1) or onehot == (0, 1, 1, 1):
            return HORIZONTAL

    @property
    def neighbor_rects(self) -> List[Rectangle]:
        return [x for x in [self.top_rect, self.left_rect, self.bot_rect, self.right_rect] if x]

    def boustrophedon_path(self, interval:float, alt=1) -> list:
        ret = []
        if self.width > self.height:
            x_min, x_max = self.lowerleft[0]+interval/2, self.upperright[0]-interval/2
            for y in np.arange(self.lowerleft[1]+interval/2, self.upperright[1], interval):
                ret.extend([(x_min, y), (x_max, y)] if alt else [(x_max, y), (x_min, y)])
                alt = not alt
        else:
            y_min, y_max = self.lowerleft[1]+interval/2, self.upperright[1]-interval/2
            for x in np.arange(self.lowerleft[0]+interval/2, self.upperright[0], interval):
                ret.extend([(x, y_min), (x, y_max)] if alt else [(x, y_max), (x, y_min)])
                alt = not alt
        return ret

    def draw(self, ax, facecolor='c', edgecolor='k', alpha=0.5) -> None:
        ax.add_patch(patch.Rectangle(
            self.lowerleft, self.width, self.height, 
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=2))


    def copy(self) -> Rectangle:
        return Rectangle(self.lowerleft, self.upperright)
    
    def merge(self, other:Rectangle) -> Rectangle|None:
        ret = None
        if self.lowerleft[0] == other.lowerleft[0] and \
           self.upperright[0] == other.upperright[0]:
            if self.lowerleft[1] + self.height == other.lowerleft[1]:
                ret = Rectangle(self.lowerleft, other.upperright)
            elif self.lowerleft[1] == other.lowerleft[1] + self.height:
                ret = Rectangle(other.lowerleft, self.upperright)

        elif self.lowerleft[1] == other.lowerleft[1] and \
           self.upperright[1] == other.upperright[1]:
            if self.lowerleft[0] + self.width == other.lowerleft[0]:
                ret = Rectangle(self.lowerleft, other.upperright)
            elif self.lowerleft[0] == other.lowerleft[0] + self.width:
                ret = Rectangle(other.lowerleft, self.upperright)
        
        return ret

    @property
    def center(self) -> tuple:
        return ((self.lowerleft[0]+self.upperright[0])/2, 
                (self.lowerleft[1]+self.upperright[1])/2)

    def grids(self, grid_size) -> List[Rectangle]:
        ret = []
        for x in np.arange(self.lowerleft[0], self.upperright[0]-grid_size/2, grid_size):
            for y in np.arange(self.lowerleft[1], self.upperright[1]-grid_size/2, grid_size):
                ret.append(Rectangle((x, y), (x+grid_size, y+grid_size)))
        return ret

    def is_adjacent(self, other:Rectangle) -> bool:
        if self.lowerleft[0] == other.lowerleft[0] and \
           self.upperright[0] == other.upperright[0]:
            if FIX(self.lowerleft[1] + self.height) == other.lowerleft[1]:
                return True
            elif self.lowerleft[1] == FIX(other.lowerleft[1] + self.height):
                return True
        elif self.lowerleft[1] == other.lowerleft[1] and \
             self.upperright[1] == other.upperright[1]:
            if FIX(self.lowerleft[0] + self.width) == other.lowerleft[0]:
                return True
            elif self.lowerleft[0] == FIX(other.lowerleft[0] + self.width):
                return True
        return False


def read_polygon(rect_dict:dict) -> Tuple[Polygon, set, set]:
    exterior, interiors, Xs, Ys = [], [], set(), set()
    for x, y in rect_dict["exterior"]:
        exterior.append((x, y))
        Xs.add(x)
        Ys.add(y)
    for interior_pts in rect_dict["interiors"]:
        interior = []
        for x, y in interior_pts:
            interior.append((x, y))
            Xs.add(x)
            Ys.add(y)
        interiors.append(interior)
    
    Xs, Ys = list(sorted(Xs)), list(sorted(Ys))
    polygon = Polygon(exterior, interiors)
    return polygon, Xs, Ys


def build_pathfinder(polygon, interval) -> Pathfinder:
    layered_isolines = LayeredIsoLines.from_polygon(polygon, interval)
    IG = IsoGraph.build(layered_isolines)
    return Pathfinder(IG, interval)
