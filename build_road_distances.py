"""Build station-to-station driving distance/time matrices from an OSM road network.

Outputs (row/column order = data/road_station_ids.npy = order of data/stations.json):
  data/road_dist_km.npy      float32 (N, N) shortest driving distance, km
  data/road_time_min.npy     float32 (N, N) free-flow driving time, minutes
  data/road_station_ids.npy  int64   (N,)   station_id per row

Matrices are directed (row = origin, col = destination) and respect one-way streets.
The downloaded drive graph is cached to data/toronto_drive.graphml.

Run with: python build_road_distances.py
"""

from pathlib import Path

import numpy as np
import osmnx as ox
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from data_loader import load_stations

DATA_DIR = Path(__file__).parent / "data"
GRAPH_PATH = DATA_DIR / "toronto_drive.graphml"
BUFFER_KM = 2.0
EARTH_RADIUS_KM = 6371.0088
# Each station is joined to its snapped road node by a straight "access" leg driven at this speed.
# Adding the legs keeps road >= haversine (triangle inequality) and separates stations sharing a node.
ACCESS_SPEED_KMH = 20.0

# The graphml file is our cache; don't let osmnx write its HTTP cache/ folder into the repo.
ox.settings.use_cache = False


def haversine_km(lat, lon):
    """Pairwise great-circle distance matrix (km) for arrays of lat/lon in degrees."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_graph(stations):
    if GRAPH_PATH.exists():
        print(f"Loading cached graph from {GRAPH_PATH}")
        return ox.load_graphml(GRAPH_PATH)

    mid_lat = np.radians(stations["lat"].mean())
    dlat = BUFFER_KM / 111.32
    dlon = BUFFER_KM / (111.32 * np.cos(mid_lat))
    bbox = (
        stations["lon"].min() - dlon,  # left
        stations["lat"].min() - dlat,  # bottom
        stations["lon"].max() + dlon,  # right
        stations["lat"].max() + dlat,  # top
    )
    print(f"Downloading drive network for bbox (W,S,E,N) = {tuple(round(v, 4) for v in bbox)} ...")
    G = ox.graph_from_bbox(bbox, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, GRAPH_PATH)
    print(f"Saved graph to {GRAPH_PATH}")
    return G


def edge_matrices(G, node_index):
    """Directed sparse adjacency matrices (length in m, travel_time in s), min over parallel edges."""
    n = len(node_index)
    best = {}  # (u, v) -> (length, travel_time)
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        key = (node_index[u], node_index[v])
        length, tt = float(d["length"]), float(d["travel_time"])
        if key in best:
            old_len, old_tt = best[key]
            best[key] = (min(old_len, length), min(old_tt, tt))
        else:
            best[key] = (length, tt)

    rows = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
    cols = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
    vals = np.array(list(best.values()), dtype=np.float64)
    # csgraph treats explicit zeros as missing edges; nudge zero-weight edges to a tiny positive value.
    vals = np.maximum(vals, 1e-6)
    W_len = sp.csr_matrix((vals[:, 0], (rows, cols)), shape=(n, n))
    W_tt = sp.csr_matrix((vals[:, 1], (rows, cols)), shape=(n, n))
    return W_len, W_tt


def main():
    stations = load_stations()
    station_ids = stations["station_id"].to_numpy(dtype=np.int64)
    lat = stations["lat"].to_numpy(dtype=np.float64)
    lon = stations["lon"].to_numpy(dtype=np.float64)
    n_st = len(stations)
    print(f"{n_st} stations")

    G = load_graph(stations)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Snap only to the largest strongly connected component so every station can reach every other.
    G = ox.truncate.largest_component(G, strongly=True)
    print(f"Largest strongly connected component: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    station_nodes, snap_m = ox.distance.nearest_nodes(G, X=lon, Y=lat, return_dist=True)
    snap_km = np.asarray(snap_m) / 1000.0
    node_ids = list(G.nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    src = np.array([node_index[nid] for nid in station_nodes], dtype=np.int64)
    print(f"Distinct snapped nodes: {len(np.unique(src))} (stations sharing a node differ only by their access legs)")

    print(f"Snap distance (m): median {np.median(snap_km) * 1000:.0f}, p95 {np.percentile(snap_km, 95) * 1000:.0f}, "
          f"max {snap_km.max() * 1000:.0f}")
    far = np.argsort(-snap_km)[:8]
    print("Largest snaps: " + ", ".join(f"{stations['name'].iat[i]} ({snap_km[i] * 1000:.0f} m)" for i in far))

    W_len, W_tt = edge_matrices(G, node_index)

    uniq, inv = np.unique(src, return_inverse=True)
    print("Running Dijkstra (length) ...")
    dist_m = dijkstra(W_len, directed=True, indices=uniq)[:, uniq][inv][:, inv]
    print("Running Dijkstra (travel_time) ...")
    time_s = dijkstra(W_tt, directed=True, indices=uniq)[:, uniq][inv][:, inv]

    access_km = snap_km[:, None] + snap_km[None, :]
    dist_km = dist_m / 1000.0 + access_km
    time_min = time_s / 60.0 + access_km / ACCESS_SPEED_KMH * 60.0
    hav = haversine_km(lat, lon)

    offdiag = ~np.eye(n_st, dtype=bool)
    bad = offdiag & ~(np.isfinite(dist_km) & np.isfinite(time_min))
    print(f"Unreachable / inf pairs: {int(bad.sum())}")

    ok = offdiag & ~bad & (hav > 0.05)  # skip near-coincident stations when computing ratios
    ratio = dist_km[ok] / hav[ok]
    med_ratio = float(np.median(ratio))
    # Speed for time fallback: median km per minute over reachable pairs.
    ok &= time_min > 0
    med_kmpm = float(np.median(dist_km[ok] / time_min[ok]))
    print(f"Road/haversine ratio: median {med_ratio:.3f}, p95 {np.percentile(ratio, 95):.3f}, "
          f"min {ratio.min():.3f}, max {ratio.max():.3f}")
    print(f"Median effective speed: {med_kmpm * 60:.1f} km/h")

    if bad.any():
        dist_km[bad] = hav[bad] * med_ratio
        time_min[bad] = dist_km[bad] / med_kmpm

    np.fill_diagonal(dist_km, 0.0)
    np.fill_diagonal(time_min, 0.0)

    below = offdiag & (dist_km < hav)
    tol_below = offdiag & (dist_km < hav - 0.2)
    print(f"Pairs with road < haversine: {int(below.sum())} "
          f"({int(tol_below.sum())} by more than 200 m) out of {int(offdiag.sum())}")

    dist_km = dist_km.astype(np.float32)
    time_min = time_min.astype(np.float32)
    np.save(DATA_DIR / "road_dist_km.npy", dist_km)
    np.save(DATA_DIR / "road_time_min.npy", time_min)
    np.save(DATA_DIR / "road_station_ids.npy", station_ids)

    print(f"road_dist_km:     {dist_km.shape} {dist_km.dtype}, mean {dist_km[offdiag].mean():.2f} km, "
          f"max {dist_km.max():.2f} km")
    print(f"road_time_min:    {time_min.shape} {time_min.dtype}, mean {time_min[offdiag].mean():.2f} min, "
          f"max {time_min.max():.2f} min")
    print(f"road_station_ids: {station_ids.shape} {station_ids.dtype}")
    asym = np.abs(dist_km - dist_km.T)[offdiag]
    print(f"Asymmetry |d(i,j)-d(j,i)|: median {np.median(asym) * 1000:.0f} m, p95 {np.percentile(asym, 95) * 1000:.0f} m")


if __name__ == "__main__":
    main()
