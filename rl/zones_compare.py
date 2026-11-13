"""Compare zone schemes: today's k-means vs road + demand, dock-balanced flow zones.

    python -m rl.zones_compare [--weights 0.3 0.5 0.7] [--out results/zones_v2]

Writes metrics.md / metrics.csv, zone_table.md (per zone, for each flow scheme) and
maps: zones_side_by_side.png and zones_<scheme>.png.

Metrics (mainland stations only; the island stations have no road link):
- docks / stations per zone: balance;
- road min within zone: mean drive time between two stations of the same zone,
  roughly the drive between stops of a multi-stop truck mission;
- profile corr within zone: mean correlation of station demand profiles (net flow
  per dock by hour, weekday + weekend) between two stations of the same zone;
- profile variance explained: share of the profile variance that the zone means
  explain (between-zone / total), i.e. how much of a station's demand pattern its
  zone predicts;
- isolated stations: none of a station's 4 nearest stations by road is in its zone.
"""
import argparse
import os

import numpy as np
import pandas as pd

from bikeshare_env.env import ISLAND_STATIONS
from rl import zones as zz
from rl.wrapper import static_data

DOWNTOWN = (-79.445, -79.335, 43.610, 43.685)     # lon min, lon max, lat min, lat max


def scheme_metrics(name, zones, T, P, cap, main):
    zof = zones.zone_of_station
    zm = zof[main]
    ks = np.unique(zm)
    docks = np.array([cap[main][zm == k].sum() for k in ks])
    n = np.array([(zm == k).sum() for k in ks])
    Tm = T[np.ix_(main, main)]
    Pm = P[main]
    Pc = Pm - Pm.mean(1, keepdims=True)
    Pc /= np.maximum(np.linalg.norm(Pc, axis=1, keepdims=True), 1e-12)
    road, corr = [], []
    for k in ks:
        m = np.flatnonzero(zm == k)
        if len(m) < 2:
            continue
        iu = np.triu_indices(len(m), 1)
        road.append(Tm[np.ix_(m, m)][iu].mean())
        corr.append((Pc[m] @ Pc[m].T)[iu].mean())
    grand = Pm.mean(0)
    sst = ((Pm - grand) ** 2).sum()
    ssb = sum((zm == k).sum() * ((Pm[zm == k].mean(0) - grand) ** 2).sum() for k in ks)
    nb = np.argsort(Tm + np.diag(np.full(len(Tm), np.inf)), axis=1)[:, :4]
    isolated = int(sum(not (zm[nb[i]] == zm[i]).any() for i in range(len(zm))))
    return {"scheme": name, "zones": int(zones.n_zones),
            "docks_min": int(docks.min()), "docks_median": int(np.median(docks)),
            "docks_max": int(docks.max()), "docks_cv": float(docks.std() / docks.mean()),
            "stations_min": int(n.min()), "stations_max": int(n.max()),
            "road_min_within_zone": float(np.mean(road)),
            "profile_corr_within_zone": float(np.mean(corr)),
            "profile_var_explained": float(ssb / sst),
            "isolated_stations": isolated}


def pattern(profile_weekday: np.ndarray) -> str:
    """Plain-words weekday pattern from a zone's net flow per dock by hour."""
    am, pm = profile_weekday[7:10].sum(), profile_weekday[16:19].sum()
    word = lambda v: "inflow" if v > 0 else "outflow"
    if max(abs(am), abs(pm)) < 0.02:
        return "balanced"
    return f"morning {word(am)} ({am:+.2f}), evening {word(pm)} ({pm:+.2f})"


def zone_table(zones, T, P, cap, main) -> pd.DataFrame:
    rows = []
    for z, idx in enumerate(zones.stations_in_zone):
        m = np.intersect1d(idx, main)
        road = T[np.ix_(m, m)][np.triu_indices(len(m), 1)].mean() if len(m) > 1 else np.nan
        w = cap[idx] / cap[idx].sum()
        prof = (P[idx] * w[:, None]).sum(0)                 # dock-weighted zone profile
        rows.append({"zone": z, "stations": len(idx), "docks": int(cap[idx].sum()),
                     "road_min_within": round(float(road), 1) if len(m) > 1 else None,
                     "weekday_pattern": pattern(prof[:24]) if len(m) else "island (no trucks)"})
    return pd.DataFrame(rows)


def draw(ax, zones, lat, lon, title, island_zone=None):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    regions = zz.zone_regions(zones, lat, lon)
    for z, geom in enumerate(regions):
        color = "#222222" if z == island_zone else cmap(z % 20)
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            x, y = g.exterior.xy
            ax.fill(x, y, color=color, alpha=0.18, lw=0)
            ax.plot(x, y, color=color, lw=0.8)
        idx = zones.stations_in_zone[z]
        ax.scatter(lon[idx], lat[idx], s=4, color=color, zorder=3)
        ax.text(lon[idx].mean(), lat[idx].mean(), str(z), fontsize=8, weight="bold",
                ha="center", va="center", zorder=4, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_title(title, fontsize=10)
    ax.set_aspect(1 / np.cos(np.radians(43.7)))
    ax.set_xticks([]), ax.set_yticks([])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--out", default=os.path.join("results", "zones_v2"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    s = static_data()
    st, ids = s["stations"], s["ids"]
    lat, lon = st["lat"].to_numpy(), st["lon"].to_numpy()
    cap = st["capacity"].to_numpy(float)
    T = s["time"].astype(float)
    T = (T + T.T) / 2
    main_idx = np.flatnonzero(~np.isin(ids, ISLAND_STATIONS))
    P = zz.demand_profiles(ids, cap)

    schemes = [("kmeans (rounds 1-4)", s["zones"], None)]
    for w in args.weights:
        schemes.append((f"flow w={w:g} ({w:.0%} road, {1 - w:.0%} demand)",
                        zz.build_flow_zones(w), zz.N_ZONES))
    rows = [scheme_metrics(name, z, T, P, cap, main_idx) for name, z, _ in schemes]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "metrics.csv"), index=False)

    lines = ["# Zone schemes compared", "",
             "Mainland stations only. Road min = mean drive time between two stations of the "
             "same zone. Profile corr = mean correlation of two same-zone stations' hourly "
             "net-flow-per-dock profiles. Variance explained = share of station profile "
             "variance captured by zone means. Isolated = none of the 4 nearest stations by "
             "road is in the same zone.", "",
             "| scheme | zones | docks min / median / max | docks CV | stations min-max | "
             "road min within zone | profile corr within zone | profile variance explained | "
             "isolated stations |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scheme']} | {r['zones']} | {r['docks_min']:,} / {r['docks_median']:,} / "
                     f"{r['docks_max']:,} | {r['docks_cv']:.2f} | {r['stations_min']}-{r['stations_max']} | "
                     f"{r['road_min_within_zone']:.1f} | {r['profile_corr_within_zone']:.2f} | "
                     f"{r['profile_var_explained']:.0%} | {r['isolated_stations']} |")
    with open(os.path.join(args.out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(args.out, "zone_table.md"), "w", encoding="utf-8") as f:
        for name, z, isl in schemes[1:]:
            t = zone_table(z, T, P, cap, main_idx)
            f.write(f"## {name}\n\n| " + " | ".join(t.columns) + " |\n|"
                    + "---|" * len(t.columns) + "\n")
            for _, r in t.iterrows():
                f.write("| " + " | ".join("–" if v is None or v != v else str(v) for v in r)
                        + " |\n")
            f.write("\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, (name, z, isl) in zip(axes.ravel(), schemes):
        draw(ax, z, lat, lon, name, isl)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "zones_side_by_side.png"), dpi=110)
    plt.close(fig)
    for name, z, isl in schemes:
        tag = name.split(" (")[0].replace(" ", "_").replace("=", "")
        for suffix, box in (("", None), ("_downtown", DOWNTOWN)):
            fig, ax = plt.subplots(figsize=(13, 9))
            draw(ax, z, lat, lon, name + (" - downtown" if box else ""), isl)
            if box:
                ax.set_xlim(box[0], box[1]), ax.set_ylim(box[2], box[3])
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"zones_{tag}{suffix}.png"), dpi=120)
            plt.close(fig)
    print("\n".join(lines[6:]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
