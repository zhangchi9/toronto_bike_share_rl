# Publishing the dashboard

Two ways to put the dashboard online. Both are **view-only**: they replay the saved runs
and never simulate, so they need no models, no PyTorch and no road network.

| | Static site on GitHub Pages | Docker Space on Hugging Face |
|---|---|---|
| Build | `python -m deploy.build_static` | `python -m deploy.build_space` |
| Output | `deploy/static_site/` (55 MB) | `deploy/hf_space/` (70 MB) |
| Server | none: HTML, JavaScript and data files | the Dash app under gunicorn |
| Cost | free | needs a paid Hugging Face plan (since 2026) |

Both build folders are ignored by this repo's git and pushed on their own.

## 1. Build every replay

```bash
python -m dashboard_data
```

Builds the replays of the 5 dashboard solutions (`rl/policy.py`, `DASHBOARD_POLICIES`).
The lookahead planner takes a few hours; the others take seconds.

## 2a. Static site on GitHub Pages (free)

```bash
python -m deploy.build_static
```

Writes `deploy/static_site/`: the page (`deploy/static/`: `index.html`, `app.js`,
`style.css`), Plotly, the truck icons and the replays as compressed per-day files
(the file layout is in `deploy/build_static.py`). It stops if any replay is missing
or out of date. The page draws the same map as the Dash app, from the same numbers.

Try it locally:

```bash
python -m http.server 8060 --directory deploy/static_site
```

then open http://localhost:8060.

**Publish (first time):** the site goes on the `gh-pages` branch of this repository,
which holds only the built site.

```bash
cd deploy/static_site
git init -b gh-pages
git add -A
git commit -m "Dashboard static site"
git remote add origin https://github.com/zhangchi9/toronto_bike_share_rl.git
git push -u origin gh-pages
```

Then on GitHub: **Settings → Pages → Build and deployment**, Source **Deploy from a
branch**, Branch **gh-pages**, folder **/ (root)**, **Save**. A minute later the site is
at https://zhangchi9.github.io/toronto_bike_share_rl/.

**Update:** rebuild (steps 1 and 2a; the build keeps the folder's `.git`), then from
`deploy/static_site`:

```bash
git add -A
git commit -m "Update dashboard"
git push
```

## 2b. Docker Space on Hugging Face (paid plan)

```bash
python -m deploy.build_space
```

Writes `deploy/hf_space/`: the app code, the saved replays, the truck routes and zone
overlay as JSON, a `Dockerfile`, `requirements.txt` and the Space's `README.md`. It
stops if any replay is missing or out of date (`--allow-missing` makes a trial build
without them).

Try it locally in view-only mode before pushing:

```bash
cd deploy/hf_space && DASHBOARD_VIEW_ONLY=1 PORT=8051 python app.py
```

Create the Space on huggingface.co (**New Space**, SDK **Docker**), then push the folder
with an access token with write permission as the password:

```bash
cd deploy/hf_space
git init -b main
git lfs install
git add .
git commit -m "Bike Share Toronto dashboard"
git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
git push -u origin main --force   # replaces the Space's starter README
```

To update, rebuild (steps 1 and 2b), then `git add -A`, `git commit` and `git push`
from `deploy/hf_space`.
