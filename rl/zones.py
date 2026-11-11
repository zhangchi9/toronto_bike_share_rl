"""Group stations into geographic zones (the RL agent acts zone-to-zone)."""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

N_ZONES = 20


@dataclass
class Zones:
    n_zones: int
    zone_of_station: np.ndarray        # (N,) int zone index per station row
    stations_in_zone: list[np.ndarray] # station row indices per zone


def make_zones(lat: np.ndarray, lon: np.ndarray, n_zones: int = N_ZONES) -> Zones:
    """K-means on (lat, lon scaled to km) so zones are compact areas."""
    xy = np.c_[np.asarray(lat) * 111.0, np.asarray(lon) * 111.0 * np.cos(np.radians(43.7))]
    labels = KMeans(n_clusters=n_zones, n_init=10, random_state=0).fit_predict(xy)
    # Order zones west -> east so indices are stable and readable.
    order = np.argsort([xy[labels == z, 1].mean() for z in range(n_zones)])
    remap = np.empty(n_zones, dtype=np.int64)
    remap[order] = np.arange(n_zones)
    zone_of = remap[labels]
    return Zones(n_zones, zone_of, [np.flatnonzero(zone_of == z) for z in range(n_zones)])


def zone_outlines(zones: Zones, lat: np.ndarray, lon: np.ndarray,
                  capacity: np.ndarray) -> list[dict]:
    """Convex hull of each zone's stations (closed ring), plus its centre and size."""
    from scipy.spatial import ConvexHull
    lat, lon = np.asarray(lat), np.asarray(lon)
    out = []
    for z, idx in enumerate(zones.stations_in_zone):
        pts = np.c_[lon[idx], lat[idx]]
        ring = pts[ConvexHull(pts).vertices] if len(idx) >= 3 else pts
        ring = np.vstack([ring, ring[:1]])
        out.append({"zone": z, "lat": ring[:, 1].round(5).tolist(),
                    "lon": ring[:, 0].round(5).tolist(),
                    "clat": float(lat[idx].mean()), "clon": float(lon[idx].mean()),
                    "stations": int(len(idx)), "docks": int(np.asarray(capacity)[idx].sum())})
    return out
