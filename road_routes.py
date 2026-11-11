"""Road geometry for drawing truck trips on the dashboard map.

Uses the same OSM drive graph as build_road_distances.py (largest strongly
connected component, fastest path by travel_time, stations snapped to their
nearest road node), so a drawn route is the route behind the env's drive time.
"""
from functools import lru_cache

import numpy as np
import osmnx as ox
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from build_road_distances import GRAPH_PATH
from data_loader import load_stations

MAX_POINTS = 150


class RoadRouter:
    def __init__(self):
        G = ox.truncate.largest_component(ox.load_graphml(GRAPH_PATH), strongly=True)
        nodes = list(G.nodes)
        index = {n: i for i, n in enumerate(nodes)}
        self.node_lat = np.array([G.nodes[n]["y"] for n in nodes])
        self.node_lon = np.array([G.nodes[n]["x"] for n in nodes])
        best = {}
        for u, v, d in G.edges(data=True):
            if u == v:
                continue
            key = (index[u], index[v])
            tt = float(d["travel_time"])
            if tt < best.get(key, np.inf):
                best[key] = tt
        rows = np.fromiter((k[0] for k in best), np.int64, len(best))
        cols = np.fromiter((k[1] for k in best), np.int64, len(best))
        vals = np.maximum(np.fromiter(best.values(), float, len(best)), 1e-6)
        self.W = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))

        st = load_stations()
        self.st_lat = st["lat"].to_numpy()
        self.st_lon = st["lon"].to_numpy()
        snapped = ox.distance.nearest_nodes(G, X=self.st_lon, Y=self.st_lat)
        self.st_node = np.array([index[n] for n in snapped])

    @lru_cache(maxsize=4096)
    def route(self, src: int, dst: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """(lats, lons) from station row `src` to station row `dst` along roads."""
        a, b = int(self.st_node[src]), int(self.st_node[dst])
        path = [b]
        if a != b:
            _, pred = dijkstra(self.W, directed=True, indices=a, return_predecessors=True)
            while path[-1] != a and pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
        path = path[::-1]
        if len(path) > MAX_POINTS:      # keep both ends, thin the middle
            keep = np.unique(np.r_[np.linspace(0, len(path) - 1, MAX_POINTS).astype(int)])
            path = [path[i] for i in keep]
        lats = [self.st_lat[src], *self.node_lat[path], self.st_lat[dst]]
        lons = [self.st_lon[src], *self.node_lon[path], self.st_lon[dst]]
        return tuple(np.round(lats, 5)), tuple(np.round(lons, 5))


@lru_cache(maxsize=1)
def get_router() -> RoadRouter:
    return RoadRouter()
