"""Group stations into zones (the RL agent acts zone-to-zone).

Two schemes:
- "kmeans" (rounds 1-4, still the default): k-means on station lat/lon, 20 zones of
  roughly equal area.
- "flow_w<w>" (built by build_flow_zones, for review): the 7 Toronto Island stations
  form their own zone (20); the 1,065 mainland stations are split into 20 zones by a
  dock-balanced k-medoids on a mix of road drive time and demand-profile distance:
      D = w * drive_time / median + (1 - w) * profile_distance / median
  Each station's profile is its average net flow per dock (returns - rentals) for
  each hour of a weekday and of a weekend / holiday, from training trips only.
  Every mainland zone holds 0.9-1.1x the mean number of docks.
"""
import json
import os
import re
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

N_ZONES = 20                     # mainland zones (k-means: all zones)
# Zones the RL agents, forecast and dashboard use by default. Rounds 1-4: "kmeans";
# rounds 5-7: "flow_w0.3" (70% demand / 30% road, 20 dock-balanced zones + island);
# round 8 on: 20 zones balanced on docks + rebalancing need, the best set for DQN with
# 1 truck. The fixed shuttle stays on k-means.
ZONE_SCHEME = "flow_w0.3_k20_need"
# Zones of the models in models/<algo>_<k>trucks/ (trained in round 5).
ROUND5_SCHEME = "flow_w0.3"
ZONES_DIR = os.path.join("models", "zones")
DOCK_BAND = 0.10                 # flow zones: docks within +-10% of the mean
FLOW_FORMAT = 1


@dataclass
class Zones:
    n_zones: int
    zone_of_station: np.ndarray        # (N,) int zone index per station row
    stations_in_zone: list[np.ndarray] # station row indices per zone


def _zones(zone_of: np.ndarray, n_zones: int) -> Zones:
    zone_of = np.asarray(zone_of, dtype=np.int64)
    return Zones(n_zones, zone_of, [np.flatnonzero(zone_of == z) for z in range(n_zones)])


def _km_xy(lat, lon) -> np.ndarray:
    return np.c_[np.asarray(lat) * 111.0, np.asarray(lon) * 111.0 * np.cos(np.radians(43.7))]


def make_zones(lat: np.ndarray, lon: np.ndarray, n_zones: int = N_ZONES) -> Zones:
    """K-means on (lat, lon scaled to km) so zones are compact areas."""
    xy = _km_xy(lat, lon)
    labels = KMeans(n_clusters=n_zones, n_init=10, random_state=0).fit_predict(xy)
    # Order zones west -> east so indices are stable and readable.
    order = np.argsort([xy[labels == z, 1].mean() for z in range(n_zones)])
    remap = np.empty(n_zones, dtype=np.int64)
    remap[order] = np.arange(n_zones)
    return _zones(remap[labels], n_zones)


# ── flow zones ───────────────────────────────────────────────────────────────
def demand_profiles(station_ids: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """(N, 48): average net flow per dock (returns - rentals) / docks per hour of day,
    24 weekday hours then 24 weekend / holiday hours, over the training period."""
    import pandas as pd
    from bikeshare_env.trips import load_period_trips
    from rl import forecast as fc
    first, last = fc.TRAIN_RANGE
    days = pd.date_range(first, last)
    off = np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
    n_days = len(days)
    trips = load_period_trips(first, last, station_ids)
    N = len(station_ids)
    flow = np.zeros((N, 2, 24))
    for when, where, sign in ((trips.start_min, trips.start_idx, -1.0),
                              (trips.end_min, trips.end_idx, 1.0)):
        keep = when < n_days * 1440
        d, m = when[keep] // 1440, when[keep] % 1440
        np.add.at(flow, (where[keep], off[d].astype(int), m // 60), sign)
    flow /= np.array([max(1, (~off).sum()), max(1, off.sum())])[None, :, None]
    flow /= np.maximum(np.asarray(capacity, float), 1)[:, None, None]
    return flow.reshape(N, 48)


def _profile_distance(P: np.ndarray) -> np.ndarray:
    sq = (P ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))


def _assign(cost: np.ndarray, docks: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Min-cost assignment of stations to zones with zone dock totals in [lo, hi]
    (LP relaxation; the few fractional stations go to their largest share)."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    N, K = cost.shape
    cols = np.arange(N * K)
    a_eq = csr_matrix((np.ones(N * K), (cols // K, cols)), shape=(N, N * K))
    cap = csr_matrix((np.tile(docks, K), (np.repeat(np.arange(K), N),
                                          (np.arange(N)[None, :] * K + np.arange(K)[:, None]).ravel())),
                     shape=(K, N * K))
    from scipy.sparse import vstack
    res = linprog(cost.ravel(), A_ub=vstack([cap, -cap]),
                  b_ub=np.r_[np.full(K, hi), np.full(K, -lo)],
                  A_eq=a_eq, b_eq=np.ones(N), bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"zone assignment LP failed: {res.message}")
    return res.x.reshape(N, K).argmax(1)


CONTIGUITY = 1.0     # weight of the gap-to-zone penalty (in road-distance units)


def _gap_to_zones(T_off: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """(N, k): drive time from each station to the nearest other member of each zone."""
    G = np.empty((len(T_off), k))
    for z in range(k):
        G[:, z] = T_off[:, labels == z].min(1)
    # A zone's only member has no other member to be near (inf above): no penalty.
    # Only seen with many zones (round 12, k=40); smaller k never hit it.
    G[np.isinf(G)] = 0.0
    return G


def _balanced_kmedoids(D: np.ndarray, docks: np.ndarray, k: int, lo: float, hi: float,
                       T_norm: np.ndarray | None = None, contiguity: float = 0.0,
                       n_starts: int = 6, max_iter: int = 30, seed: int = 0):
    """k-medoids on D with zone dock totals in [lo, hi]. With `contiguity` > 0, each
    assignment also pays contiguity x (normalised drive time to the zone's nearest
    member from the previous step), which keeps zones in one piece."""
    rng = np.random.default_rng(seed)
    N = len(D)
    T_off = None if T_norm is None else T_norm + np.diag(np.full(N, np.inf))
    best = (np.inf, None)
    for _ in range(n_starts):
        med = [int(rng.integers(N))]                          # k-means++ style seeding
        for _ in range(k - 1):
            d2 = D[:, med].min(1) ** 2
            med.append(int(rng.choice(N, p=d2 / d2.sum())))
        labels, seen = None, set()
        for _ in range(max_iter):
            cost = D[:, med].copy()
            if contiguity and labels is not None:
                cost += contiguity * _gap_to_zones(T_off, labels, k)
            new = _assign(cost, docks, lo, hi)
            key = new.tobytes()
            if key in seen:                    # converged (or cycling between two states)
                labels = new
                break
            seen.add(key)
            labels = new
            for z in range(k):
                m = np.flatnonzero(labels == z)
                med[z] = int(m[np.argmin(D[np.ix_(m, m)].sum(1))])
        total = D[np.arange(N), np.asarray(med)[labels]].sum()
        if T_off is not None and contiguity:
            G = _gap_to_zones(T_off, labels, k)
            total += contiguity * G[np.arange(N), labels].sum()
        if total < best[0]:
            best = (total, labels.copy())
    return best[1]


STRAY_MIN = 4.0      # a station this far (drive minutes) from every zone-mate is a stray


def _strays(labels, T, nb):
    """(isolated, far): no zone-mate among the nearest stations / the nearest
    zone-mate is more than STRAY_MIN minutes away."""
    isolated, far = [], []
    for i in range(len(labels)):
        if not (labels[nb[i]] == labels[i]).any():
            isolated.append(i)
            same = np.flatnonzero(labels == labels[i])
            same = same[same != i]
            if not len(same) or T[i, same].min() > STRAY_MIN:
                far.append(i)
    return isolated, far


def _fix_isolated(labels, T, docks, lo, hi, n_nb: int = 4, passes: int = 5):
    """Move each station none of whose n_nb nearest stations by road is in its zone
    to the majority zone of those neighbours, if both zones stay within the dock band
    (2% slack). Repeats until nothing moves. Returns (labels, isolated_before, moved)."""
    labels = labels.copy()
    nb = np.argsort(T + np.diag(np.full(len(T), np.inf)), axis=1)[:, :n_nb]
    totals = np.bincount(labels, weights=docks)
    n_before = len(_strays(labels, T, nb)[0])
    moved = 0
    for _ in range(passes):
        isolated, _far = _strays(labels, T, nb)
        changed = 0
        for i in isolated:
            if (labels[nb[i]] == labels[i]).any():    # an earlier move fixed it
                continue
            to = np.bincount(labels[nb[i]]).argmax()
            src, c = labels[i], docks[i]
            if totals[to] + c <= 1.02 * hi and totals[src] - c >= 0.98 * lo:
                labels[i] = to
                totals[to] += c
                totals[src] -= c
                changed += 1
        moved += changed
        if not changed:
            break
    return labels, n_before, moved


BALANCES = ("docks", "need")
_SCHEME_RE = re.compile(r"flow_w([0-9.]+)(?:_k(\d+))?(?:_(docks|need))?$")


def flow_scheme(w: float, k: int = N_ZONES, balance: str = "docks") -> str:
    """Scheme name: "flow_w0.3" for 20 dock-balanced zones (rounds 5-7), else e.g.
    "flow_w0.3_k30_need"."""
    if k == N_ZONES and balance == "docks":
        return f"flow_w{w:g}"
    return f"flow_w{w:g}_k{k}_{balance}"


def parse_flow_scheme(scheme: str) -> tuple[float, int, str]:
    m = _SCHEME_RE.match(scheme)
    if not m:
        raise ValueError(f"unknown zone scheme {scheme!r}")
    return float(m.group(1)), int(m.group(2) or N_ZONES), m.group(3) or "docks"


NEED_FILE = "station_need.npz"


def station_need(verbose: bool = True) -> np.ndarray:
    """(N,) rebalancing need per station row: lost requests (empty station) plus
    blocked returns (full station) in one no-truck run over the training period
    (May 1 - Jul 17, bikes carry over). Cached in models/zones/station_need.npz."""
    from rl import forecast as fc
    from rl.wrapper import static_data
    ids = static_data()["ids"]
    path = os.path.join(ZONES_DIR, NEED_FILE)
    if os.path.exists(path):
        with np.load(path) as f:
            if np.array_equal(f["station_ids"], ids):
                return f["need"]
    from bikeshare_env.rollout import simulate_period
    r = simulate_period(*fc.TRAIN_RANGE, "no_rebalancing")
    t = r.trips
    if r.bikes.shape[1] != len(ids):
        raise ValueError("simulator station list does not match static_data")
    N, T = len(ids), len(r.bikes)
    cap = static_data()["stations"]["capacity"].to_numpy()
    lost = np.bincount(t.start_idx[~r.trip_started], minlength=N)
    ok = r.trip_started & (t.end_min >= 1) & (t.end_min <= T)
    e_min, e_idx = t.end_min[ok], t.end_idx[ok]
    full = r.bikes[e_min - 1, e_idx] >= cap[e_idx]
    blocked = np.bincount(e_idx[full], minlength=N)
    need = (lost + blocked).astype(float)
    os.makedirs(ZONES_DIR, exist_ok=True)
    np.savez(path, need=need, lost=lost, blocked=blocked, station_ids=ids)
    if verbose:
        print(f"  station need: {lost.sum():,} lost + {blocked.sum():,} blocked returns")
    return need


def zone_weights(balance: str, cap: np.ndarray) -> np.ndarray:
    """(N,) what the zone band balances: docks, or half docks share + half need share
    (rescaled to dock units, so the +-10% band means the same thing)."""
    cap = np.asarray(cap, float)
    if balance == "docks":
        return cap
    from bikeshare_env.env import ISLAND_STATIONS
    from rl.wrapper import static_data
    need = station_need(verbose=False)
    main = ~np.isin(static_data()["ids"], ISLAND_STATIONS)
    share = 0.5 * cap / cap[main].sum() + 0.5 * need / max(need[main].sum(), 1.0)
    return share * cap[main].sum()


def build_flow_zones(w: float = 0.5, verbose: bool = True, k: int = N_ZONES,
                     balance: str = "docks") -> Zones:
    """Island zone (zone k) + k balanced mainland zones; saved to models/zones/.
    balance="docks": each zone holds 0.9-1.1x the mean docks. balance="need": the
    band applies to a weight that is half docks share, half rebalancing-need share
    (station_need), so areas that run empty or full get smaller zones."""
    if balance not in BALANCES:
        raise ValueError(f"balance must be one of {BALANCES}")
    from bikeshare_env.env import ISLAND_STATIONS
    from rl.wrapper import static_data
    s = static_data()
    st, ids = s["stations"], s["ids"]
    lat, lon = st["lat"].to_numpy(), st["lon"].to_numpy()
    cap = st["capacity"].to_numpy(float)
    island = np.isin(ids, ISLAND_STATIONS)
    main = np.flatnonzero(~island)

    T = s["time"].astype(float)
    T = (T + T.T) / 2
    Tm = T[np.ix_(main, main)]
    P = demand_profiles(ids, cap)[main]
    Dp = _profile_distance(P)
    off = ~np.eye(len(main), dtype=bool)
    D = w * Tm / np.median(Tm[off]) + (1 - w) * Dp / np.median(Dp[off])

    weight = zone_weights(balance, cap)[main]
    mean = weight.sum() / k
    lo, hi = (1 - DOCK_BAND) * mean, (1 + DOCK_BAND) * mean
    labels = _balanced_kmedoids(D, weight, k, lo, hi,
                                T_norm=Tm / np.median(Tm[off]), contiguity=CONTIGUITY)
    labels, n_isolated, moved = _fix_isolated(labels, Tm, weight, lo, hi)

    xy = _km_xy(lat, lon)[main]          # number mainland zones west -> east
    order = np.argsort([xy[labels == z, 1].mean() for z in range(k)])
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    zone_of = np.full(len(ids), k, dtype=np.int64)            # island = zone k
    zone_of[main] = remap[labels]

    name = flow_scheme(w, k, balance)
    _loaded.pop(name, None)
    os.makedirs(ZONES_DIR, exist_ok=True)
    meta = {"format": FLOW_FORMAT, "w": w, "k": k, "balance": balance,
            "dock_band": DOCK_BAND, "isolated_before_fix": n_isolated, "moved_by_fix": moved}
    np.savez(os.path.join(ZONES_DIR, f"{name}.npz"), zone_of_station=zone_of,
             station_ids=ids, meta=np.array(json.dumps(meta)))
    if verbose:
        d = np.bincount(zone_of, weights=cap)[:k]
        print(f"  {name}: docks per zone {d.min():.0f}-{d.max():.0f} "
              f"(mean {cap[main].sum() / k:.0f}); isolated {n_isolated}, moved {moved}")
    return _zones(zone_of, k + 1)


_loaded: dict[str, Zones] = {}


def load_zones(scheme: str | None = None) -> Zones:
    """Zones for a scheme: "kmeans" or "flow_w<w>" (built on first use). Default:
    ZONE_SCHEME at call time."""
    scheme = scheme or ZONE_SCHEME
    if scheme in _loaded:
        return _loaded[scheme]
    from rl.wrapper import static_data
    s = static_data()
    if scheme == "kmeans":
        st = s["stations"]
        z = make_zones(st["lat"].to_numpy(), st["lon"].to_numpy())
    elif scheme.startswith("flow_w"):
        w, k, balance = parse_flow_scheme(scheme)
        path = os.path.join(ZONES_DIR, f"{flow_scheme(w, k, balance)}.npz")
        if not os.path.exists(path):
            z = build_flow_zones(w, verbose=False, k=k, balance=balance)
        else:
            with np.load(path) as f:
                if not np.array_equal(f["station_ids"], s["ids"]):
                    raise ValueError(f"{path} was built for another station list; "
                                     "delete it to rebuild")
                zone_of = f["zone_of_station"]
                z = _zones(zone_of, int(zone_of.max()) + 1)
    else:
        raise ValueError(f"unknown zone scheme {scheme!r}")
    _loaded[scheme] = z
    return z


# ── map outlines ─────────────────────────────────────────────────────────────
def zone_outlines(zones: Zones, lat: np.ndarray, lon: np.ndarray,
                  capacity: np.ndarray) -> list[dict]:
    """Each zone's shape for the dashboard overlay: one outer ring per zone (see
    zone_display_shapes), a label point inside it, and the zone's real size."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    out = []
    for z, poly in enumerate(zone_display_shapes(zones, lat, lon)):
        idx = zones.stations_in_zone[z]
        ring = np.asarray(poly.simplify(0.0003).exterior.coords)
        c = poly.centroid if poly.contains(poly.centroid) else poly.representative_point()
        out.append({"zone": z, "lat": ring[:, 1].round(5).tolist(),
                    "lon": ring[:, 0].round(5).tolist(),
                    "clat": round(float(c.y), 5), "clon": round(float(c.x), 5),
                    "stations": int(len(idx)), "docks": int(np.asarray(capacity)[idx].sum())})
    return out


def island_zone(zones: Zones) -> int | None:
    """The zone made of exactly the Toronto Island stations, if there is one."""
    from bikeshare_env.env import ISLAND_STATIONS
    from rl.wrapper import static_data
    island = np.isin(static_data()["ids"], ISLAND_STATIONS)
    if len(island) != len(zones.zone_of_station):
        return None
    zs = np.unique(zones.zone_of_station[island])
    if len(zs) == 1 and np.array_equal(np.flatnonzero(zones.zone_of_station == zs[0]),
                                       np.flatnonzero(island)):
        return int(zs[0])
    return None


def zone_display_shapes(zones: Zones, lat: np.ndarray, lon: np.ndarray,
                        margin_km: float = 0.3, gap_km: float = 1.5) -> list:
    """One polygon per zone, for drawing only. Zones built partly from demand are not
    always one connected area: a few stations can sit in a patch surrounded by other
    zones. Here each zone keeps its largest connected area, and every detached patch
    is drawn as part of the neighbouring zone it shares the longest border with. The
    shapes tile the station area without overlaps or gaps (gaps under 2 * gap_km
    between stations are filled; the outer edge is margin_km beyond the stations).
    Station membership is unchanged; only the drawing simplifies it."""
    import shapely
    from shapely.ops import unary_union
    kx = 111.0 * np.cos(np.radians(43.7))
    x, y = np.asarray(lon) * kx, np.asarray(lat) * 111.0
    pts = shapely.points(x, y)
    cover = unary_union(shapely.buffer(pts, gap_km)).buffer(-(gap_km - margin_km))
    cover = unary_union([shapely.Polygon(p.exterior) for p in shapely.get_parts(cover)])  # no holes
    box = shapely.box(x.min() - 5, y.min() - 5, x.max() + 5, y.max() + 5)
    cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(pts), extend_to=box))
    owner = np.full(len(pts), -1)              # cell index of each station
    tree = shapely.STRtree(cells)
    for i, p in enumerate(pts):
        owner[i] = tree.query(p, predicate="within")[0]
    shape = {z: unary_union(cells[owner[idx]]).intersection(cover)
             for z, idx in enumerate(zones.stations_in_zone)}

    def parts(g):
        return sorted((p for p in shapely.get_parts(g) if p.geom_type == "Polygon"),
                      key=lambda p: -p.area)
    main = {z: parts(g)[0] for z, g in shape.items()}
    patches = [p for z, g in shape.items() for p in parts(g)[1:]]
    # The island zone (flow schemes) is drawn around its own 7 stations only:
    # mainland patches are never added to it.
    isl = island_zone(zones)
    takers = [z for z in main if z != isl]
    for _ in range(20):                         # a patch may only touch other patches
        left = []
        for p in patches:
            border = {z: p.boundary.intersection(main[z].buffer(1e-6)).length for z in takers}
            z = max(border, key=border.get)
            if border[z] > 0:
                main[z] = unary_union([main[z], p.buffer(1e-6)])
                main[z] = parts(main[z])[0]
            else:
                left.append(p)
        if not left:
            break
        patches = left
    to_lonlat = np.array([kx, 111.0])
    return [shapely.transform(main[z], lambda a: a / to_lonlat) for z in range(len(main))]


def zone_regions(zones: Zones, lat: np.ndarray, lon: np.ndarray, buffer_km: float = 0.3):
    """Non-overlapping zone shapes: the union of each zone's station Voronoi cells,
    clipped to the area within buffer_km of a station. Returns shapely geometries in
    (lon, lat) per zone (for maps; zones that aren't round need this, not hulls)."""
    import shapely
    from shapely.ops import unary_union
    kx = 111.0 * np.cos(np.radians(43.7))
    x, y = np.asarray(lon) * kx, np.asarray(lat) * 111.0
    pts = shapely.points(x, y)
    cover = unary_union(shapely.buffer(pts, buffer_km))
    cells = shapely.voronoi_polygons(shapely.multipoints(pts), extend_to=cover)
    cells = list(shapely.get_parts(cells))
    # Match each cell to the station inside it.
    owner = np.full(len(cells), -1)
    tree = shapely.STRtree(pts)
    for c_i, cell in enumerate(cells):
        hit = tree.query(cell, predicate="contains")
        if len(hit):
            owner[c_i] = hit[0]
    out = []
    for z, idx in enumerate(zones.stations_in_zone):
        mine = set(idx.tolist())
        shape = unary_union([cells[c] for c in range(len(cells)) if owner[c] in mine])
        shape = shapely.intersection(shape, cover)
        out.append(shapely.transform(shape, lambda a: a / np.array([kx, 111.0])))
    return out
