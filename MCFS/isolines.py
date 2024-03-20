from __future__ import annotations
from typing import List, Tuple
from enum import IntEnum
from collections import defaultdict

import math
import numpy as np
from skimage import measure

from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.coords import CoordinateSequence


class IsoLine:

    def __init__(self, polylines, layer:int, inner_idx:int, interp_dist:float=0) -> None:
        self.layer = layer
        self.inner_idx = inner_idx
        self.line = LineString(polylines)
        if interp_dist > 0:
            n_pts = int(self.line.length/interp_dist)
            new_points = [self.line.interpolate(i/float(n_pts - 1), normalized=True) for i in range(n_pts)]
            self.line = LineString(new_points)
    
    @property
    def index(self) -> tuple:
        return (self.layer, self.inner_idx)
    
    def __len__(self) -> int:
        return len(self.line.coords)

    def __hash__(self) -> int:
        return self.index.__hash__()

    def __eq__(self, other:IsoLine) -> bool:
        return (self.layer == other.layer) and (self.inner_idx == other.inner_idx)
    
    def __ne__(self, other:IsoLine) -> bool:
        return not self == other
    
    def __lt__(self, other:IsoLine) -> bool:
        return (self.layer < other.layer) or \
               (self.layer == other.layer and self.inner_idx < other.inner_idx) 

    @property
    def coords(self) -> CoordinateSequence:
        return self.line.coords

    def segment(self, st_idx:int, ed_idx:int=None) -> list:
        if ed_idx is None:
            return self.coords[st_idx:-1] + self.coords[:st_idx]

        if st_idx <= ed_idx:
            return self.coords[st_idx:(ed_idx+1)%len(self)]
        else:
            return self.coords[st_idx:-1] + self.coords[:(ed_idx+1)%len(self)]

    def nearest(self, p:tuple) -> Tuple[float, int, tuple]:
        min_dist_pair = (float('inf'), None, None)
        for i, pt in enumerate(self.coords):
            dist = math.hypot(pt[0]-p[0], pt[1]-p[1])        
            if dist < min_dist_pair[0]:
                min_dist_pair = (dist, i, (pt[0], pt[1]))

        return min_dist_pair

    def A(self, p_idx:int, offset=1) -> Tuple[int, tuple]:
        Ap_idx = (p_idx + offset) % (len(self) - 1)
        return Ap_idx, self.coords[Ap_idx]

    def B(self, p_idx:int, offset=1) -> Tuple[int, tuple]:
        Bp_idx = (p_idx - offset) % (len(self) - 1)
        return Bp_idx, self.coords[Bp_idx]


class LayeredIsoLines:

    class NodeType(IntEnum):
        Region = 0,
        Bridge = 1,

    def __init__(self) -> None:
        self.dat:List[List[IsoLine]] = []
        self.O = defaultdict(set)
        self.P = defaultdict(set)
    
    @staticmethod
    def from_polygon(polygon:Polygon, interval=0.1) -> LayeredIsoLines:
        li = LayeredIsoLines()

        min_x, min_y, max_x, max_y = polygon.bounds
        x_points, y_points = np.meshgrid(
            np.arange(min_x, max_x, interval),
            np.arange(min_y, max_y, interval)
        )
        
        # polygon boundary
        boundaries = MultiLineString([polygon.exterior])
        for interior in polygon.interiors:
            boundaries = boundaries.union(interior)

        # set iso-value to the distance to the boundary
        isovalues = np.zeros_like(x_points)
        for i in range(x_points.shape[0]):
            for j in range(y_points.shape[1]):
                isovalues[i, j] = Point((x_points[i, j], y_points[i, j])).distance(boundaries)

        # isolines
        scaler = np.array([[max_y-min_y, max_x-min_x]]) / np.reshape(x_points.shape, (1, 2))
        offset = np.array([[min_y, min_x]])
        for layer, level in enumerate(np.arange(np.min(isovalues), np.max(isovalues), interval)):
            lines = []
            for contour in measure.find_contours(isovalues, level):
                contour = np.fliplr(scaler * contour + offset)
                if polygon.contains(Point(contour[0])) and contour.shape[0] > 5:
                    lines.append(IsoLine(contour, layer=layer-1, inner_idx=len(lines), interp_dist=interval))
            if lines != []:
                li.dat.append(lines)
        
        return li
    
    @property
    def n_layers(self) -> int:
        return len(self.dat)
    
    def at(self, layer_idx:int, inner_idx:int=None) -> IsoLine|List[IsoLine]:
        if layer_idx < 0 or layer_idx >= self.n_layers:
            return []
        if inner_idx is not None:
            return self.dat[layer_idx][inner_idx]
        else:
            return self.dat[layer_idx]
 
    def draw_isolines(self, ax) -> None:
        for layer in range(self.n_layers):
            for i in range(len(self.at(layer))):
                isoline = self.at(layer, i)
                ax.plot(isoline.coords.xy[0], isoline.coords.xy[1], 'k')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

