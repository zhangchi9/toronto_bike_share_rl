"""Cached station-to-station driving distance / time matrices.

Built once by `python build_road_distances.py` (OSMnx + Dijkstra); the env only
loads the .npy files here.
"""
import os

import numpy as np

DIST_PATH = "data/road_dist_km.npy"
TIME_PATH = "data/road_time_min.npy"
IDS_PATH = "data/road_station_ids.npy"


def load_road_matrices(station_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dist_km, time_min), rows/cols aligned to `station_ids`."""
    missing = [p for p in (DIST_PATH, TIME_PATH, IDS_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Road distance matrices not found ({', '.join(missing)}). "
            "Run `python build_road_distances.py` once to build them."
        )
    dist = np.load(DIST_PATH)
    time = np.load(TIME_PATH)
    ids = np.load(IDS_PATH)
    pos = {int(s): i for i, s in enumerate(ids)}
    try:
        order = np.array([pos[int(s)] for s in station_ids])
    except KeyError as e:
        raise ValueError(
            f"Station {e.args[0]} missing from road matrices; "
            "re-run build_road_distances.py after fetch_stations.py."
        ) from None
    return dist[np.ix_(order, order)], time[np.ix_(order, order)]


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """All-pairs great-circle distance in km (straight line, not road)."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = (np.sin((la - la.T) / 2) ** 2
         + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
