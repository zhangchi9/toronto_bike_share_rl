# Deploying the dashboard to a Hugging Face Space

The public dashboard is a **view-only** copy: it replays the saved runs and never
simulates, so it needs no models, no PyTorch and no road network. It runs as a
Docker Space on the free CPU hardware.

## 1. Build every replay

```bash
python -m dashboard_data
```

Builds the replays of the 5 dashboard solutions (`rl/policy.py`, `DASHBOARD_POLICIES`).
The lookahead planner takes a few hours; the others take seconds.

## 2. Build the Space folder

```bash
python -m deploy.build_space
```

Writes `deploy/hf_space/` (ignored by this repo's git): the app code, the saved
replays, the truck routes and zone overlay as JSON, a `Dockerfile`, `requirements.txt`
and the Space's `README.md`. It stops if any replay is missing or out of date
(`--allow-missing` makes a trial build without them).

Try it locally in view-only mode before pushing:

```bash
cd deploy/hf_space && DASHBOARD_VIEW_ONLY=1 PORT=8051 python app.py
```

## 3. Create the Space and push (once)

1. On huggingface.co: **New Space**, SDK **Docker**, hardware **CPU basic (free)**,
   public or private.
2. Install Git LFS once (`git lfs install`). The replays are `.npz` files of about
   10 MB each, which a Space only accepts through LFS (`.gitattributes` handles it).
3. Push the folder, using an access token with write permission as the password:

```bash
cd deploy/hf_space
git init -b main
git lfs install
git add .
git commit -m "Bike Share Toronto dashboard"
git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
git push -u origin main --force   # replaces the Space's starter README
```

The Space builds the Docker image (a few minutes) and serves the dashboard at
`https://<your-username>-<space-name>.hf.space`.

## Updating

Rebuild the replays and the folder (steps 1 and 2); `build_space` keeps the folder's
`.git`. Then, from `deploy/hf_space`:

```bash
git add -A
git commit -m "Update replays"
git push
```
