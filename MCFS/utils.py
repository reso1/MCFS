from typing import List
from functools import wraps
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy import interpolate
from shapely.geometry import LineString, Polygon


def timeit(func):
    """ https://gist.github.com/rivergold/df4e77f1322cf4cb85910735f437059f """
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"@timefn: {func.__name__} took {end_time - start_time:.3f} seconds.")
        return result
    return measure_time


def segment(pi:np.ndarray, st_idx:int, ed_idx:int) -> np.ndarray:
    if st_idx <= ed_idx:
        return pi[st_idx:ed_idx+1]
    else:
        return np.vstack([pi[st_idx:], pi[:ed_idx+1]])


def path_length(pi:list) -> float:
    return sum([np.hypot(pi[i][0]-pi[i+1][0], pi[i][1]-pi[i+1][1]) for i in range(len(pi)-1)])


def curvature(polyline:np.ndarray) -> float:    
    dx, dy = np.gradient(polyline[:, 0]), np.gradient(polyline[:, 1])
    d2x, d2y = np.gradient(dx), np.gradient(dy)
    k = (dx * d2y - d2x * dy) ** 2 / (dx * dx + dy * dy) ** 3
    return k


def remove_pts_repetition(pi:list):
    j = 0
    while j < len(pi) - 1:
        if pi[j] == pi[j+1]:
            pi.pop(j)
        else:
            j += 1


def spline_interpolate(polylines:list, interval:float=0.1) -> list:
    x, y = zip(*polylines)
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    f, u = interpolate.splprep([x, y], s=0, per=True)
    xint, yint = interpolate.splev(
        np.linspace(0, 1, int(LineString(polylines).length // interval)), f)
    return np.vstack([xint, yint]).T.tolist()


def linear_interpolate(polylines:list, interval:float=0.1) -> list:
    if polylines[-1] != polylines[0]:
        line = LineString(polylines + [polylines[0]])
    else:
        line = LineString(polylines)
    n_pts = int(line.length/interval)
    new_points = [line.interpolate(i/float(n_pts - 1), normalized=True) for i in range(n_pts)]
    return LineString(new_points)


def draw_polygon(polygon:Polygon, ax, color='k') -> None:
    ax.fill(polygon.exterior.xy[0], polygon.exterior.xy[1], color=color, alpha=0.2)
    for interior in polygon.interiors:
        ax.fill(interior.xy[0], interior.xy[1], color='white')


def draw_path(pi:list, ax, color='k') -> None:
    x, y = zip(*pi)
    ax.plot(x[0], y[0], '^k', ms=2)
    ax.plot(x[-1], y[-1], 'ok', ms=2)
    ax.plot(x, y, f"-{color}")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)


def draw_pockets(pockets, ax) -> None:
    colors = ["r", 'g', 'b', 'k', 'c']
    for i, p in enumerate(pockets):
        for j, line in enumerate(p.isolines):
            coords = np.array(line.coords)
            ax.plot(coords[:, 0], coords[:, 1], f"{colors[i%5]}")
            ax.axis("equal")


def draw_isograph(isograph:nx.Graph, ax, node_color='c', edge_label=True, R=None) -> None:
    pos = {node:(node[1], node[0]) for node in isograph.nodes}
    node_weights = [isograph.nodes[node]["weight"] for node in isograph.nodes]
    max_nw, min_nw = max(node_weights), min(node_weights)
    node_size = [100*(1 + (nw-min_nw)/(max_nw-min_nw)) for nw in node_weights]
    labels = {k:-l for k, l in nx.get_edge_attributes(isograph,'weight').items()}
    nx.draw(isograph, pos, ax, with_labels=False, node_size=node_size, node_color=node_color, alpha=0.6, font_size=8)
    nx.draw_networkx_labels(isograph, pos, ax=ax, labels=nx.get_node_attributes(isograph,'weight'), font_size=8)
    if edge_label:
        nx.draw_networkx_edge_labels(isograph, pos, ax=ax, edge_labels=labels, font_size=8)
    if R:
        r_pos = pos[R]
        ax.plot(r_pos[0], r_pos[1], f'{node_color}p', ms=20)


def draw_isograph_hypergraphs_node_labels(isograph:nx.Graph, hypG:nx.Graph) -> None:
    fig, ax = plt.subplots(1, 3)
    pos = {node:(node[1], node[0]) for node in isograph.nodes}
    nx.draw(isograph, pos, ax[0], with_labels=True, node_size=200, node_color='lightblue', font_size=8)
    
    isotree = nx.minimum_spanning_tree(isograph)
    node_colors = []
    for node in isotree.nodes:
        node_colors.append("lightblue" if nx.degree(isotree, node)<=2 else "red")
    nx.draw(isotree, pos, ax[1], with_labels=True, node_size=200, node_color='lightblue', font_size=8)

    pos, node_colors = {}, []
    for i, node in enumerate(hypG.nodes):
        sum_x, sum_y = 0, 0
        for v in hypG.nodes[node]["V_iso"]:
            sum_x += v[1]
            sum_y += v[0]
        pos[node] = (sum_x/len(hypG.nodes[node]["V_iso"]),
                     sum_y/len(hypG.nodes[node]["V_iso"])+ (0.5 if (sum_x/len(hypG.nodes[node]["V_iso"]))%2==1 else 0))
        node_colors.append("lightblue" if node[0]=="R" else "red")

    nx.draw(hypG, pos, ax[2], with_labels=True, node_size=500, node_color=node_colors, font_size=16)


def draw_MCPP_solution(Pi):
    colors = ["r", "b", "k", "c", "g", "m"]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i, pi in enumerate(Pi):
        ax.plot(pi[:, 0], pi[:, 1], f'-{colors[i%len(colors)]}')
        ax.axis("equal")

    return fig, ax


class LineDataUnits(Line2D):
    """
    https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1) 
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)

