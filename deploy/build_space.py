"""Build a self-contained Hugging Face Space for the dashboard (view-only).

    python -m deploy.build_space                  # -> deploy/hf_space/
    python -m deploy.build_space --allow-missing  # trial build before every replay exists

The Space replays the saved runs only (dashboard_data's view-only mode): no
simulation, no PyTorch, no models, no road network. It holds

    app.py, dashboard_data.py, data_loader.py, bikeshare_env/, assets/
    data/stations.json
    data/replay/<policy>.npz, _trips.npz      the saved replays (Git LFS)
    data/replay/manifest.json                 policy key -> dropdown label
    data/replay/_routes.json                  road route of every truck leg
    data/replay/zones.json                    the zone overlay
    Dockerfile, requirements.txt, README.md (Space settings), .gitattributes

The policies are the dashboard's (no trucks + rl.policy.DASHBOARD_POLICIES). Every
replay must be built and up to date (python -m dashboard_data) unless
--allow-missing is given; missing ones are left out of the Space. See
deploy/README.md for pushing the folder to a Space.
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hf_space")
CODE = ["app.py", "dashboard_data.py", "data_loader.py"]
PACKAGES = ["bikeshare_env", "assets"]
PORT = 7860             # what a Docker Space expects

REQUIREMENTS = """\
dash==4.4.1
pandas==2.3.3
numpy==2.2.6
gymnasium==1.4.0
duckdb==1.5.6
gunicorn==23.0.0
"""

DOCKERFILE = f"""\
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH DASHBOARD_VIEW_ONLY=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE {PORT}
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:{PORT}", "--workers", "1", "--threads", "8", "--timeout", "180"]
"""

README = f"""\
---
title: Bike Share Toronto Rebalancing
emoji: 🚲
colorFrom: blue
colorTo: green
sdk: docker
app_port: {PORT}
pinned: false
---

# Bike Share Toronto: truck rebalancing replays

Minute-by-minute replays of Bike Share Toronto, Aug 1 - Sep 30, 2026, simulated from
real trips, under {{n}} rebalancing solutions:

{{rows}}

Each solution is one continuous simulated run: stations start 35% full at 00:00 on
Aug 1 and bikes carry over between days. Pick a date range and a solution to watch
station fill levels, rides and trucks.
"""

GITATTRIBUTES = """\
*.npz filter=lfs diff=lfs merge=lfs -text
data/replay/*.json filter=lfs diff=lfs merge=lfs -text
"""


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--allow-missing", action="store_true",
                    help="leave out replays that aren't built yet (for a trial build)")
    args = ap.parse_args()

    import dashboard_data as dd
    import rl.policy as rp
    if dd.VIEW_ONLY:
        raise SystemExit("unset DASHBOARD_VIEW_ONLY: the build reads the full project")
    keys = ["no_rebalancing", *rp.DASHBOARD_POLICIES]
    missing = [k for k in keys if k not in dd.POLICIES or not dd.is_current(k)]
    if missing and not args.allow_missing:
        raise SystemExit(f"replays not built or out of date: {missing}; run "
                         "python -m dashboard_data (or pass --allow-missing for a trial)")
    keys = [k for k in keys if k not in missing]
    labels = {k: dd.POLICIES[k].label for k in keys}

    out = args.out
    if os.path.exists(out):             # start clean, but keep the Space's git history
        for name in os.listdir(out):
            if name != ".git":
                p = os.path.join(out, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    rep = os.path.join(out, "data", "replay")
    os.makedirs(rep)
    for f in CODE:
        shutil.copy2(f, out)
    for d in PACKAGES:
        shutil.copytree(d, os.path.join(out, d), ignore=_ignore)
    shutil.copy2(os.path.join("data", "stations.json"), os.path.join(out, "data"))
    for k in ["_trips", *keys]:
        shutil.copy2(os.path.join(dd.REPLAY_DIR, f"{k}.npz"), rep)

    def dump(name, obj):
        with open(os.path.join(rep, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))

    dump("manifest.json", {"policies": labels})
    print("road routes of every truck leg ...", flush=True)
    dump("_routes.json", dd.leg_routes(keys))
    import app                                  # the zone overlay, as the full app draws it
    dump("zones.json", app.zone_store())

    rows = "\n".join(f"- {label}" for label in labels.values())
    files = {"requirements.txt": REQUIREMENTS, "Dockerfile": DOCKERFILE,
             "README.md": README.format(n=len(labels), rows=rows),
             ".gitattributes": GITATTRIBUTES}
    for name, text in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    print(f"wrote {out}: {len(keys)} solutions ({', '.join(keys)}), {size / 1e6:.0f} MB")
    if missing:
        print(f"left out (not built yet): {missing}")


if __name__ == "__main__":
    main()
