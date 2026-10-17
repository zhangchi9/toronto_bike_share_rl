"""
Bike Share Toronto — simulated day viewer
Runs the BikeShareToronto-v0 env for the chosen date + rebalancing policy, then
animates the recorded day: station fill levels, active rides and live metrics.
Usage:
    python fetch_stations.py         # once
    python preprocess.py             # once
    python build_road_distances.py   # once
    python app.py                    # http://localhost:8050
"""
import base64
from functools import lru_cache

from dash import Dash, dcc, html, Input, Output, State, no_update

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
from bikeshare_env.policies import POLICIES
from bikeshare_env.rollout import simulate_day
from data_loader import load_stations

# ── Globals ───────────────────────────────────────────────────────────────────
STATIONS_DF = load_stations()

TORONTO_CENTER = {"lat": 43.655, "lon": -79.38}
MIN_DATE = "2026-01-01"
MAX_DATE = "2026-09-30"
DEFAULT_DATE = "2026-06-15"
DEFAULT_POLICY = "no_rebalancing"
INIT_OCCUPANCY = DEFAULT_INIT_OCCUPANCY

_STATION_STORE = {
    "lat": STATIONS_DF["lat"].tolist(),
    "lon": STATIONS_DF["lon"].tolist(),
    "name": STATIONS_DF["name"].tolist(),
    "cap":  STATIONS_DF["capacity"].tolist(),
}


@lru_cache(maxsize=4)
def get_simulation(date_str: str, policy: str) -> dict:
    """Simulate one day and serialize it for the browser (~3 MB)."""
    r = simulate_day(date_str, policy, init_occupancy=INIT_OCCUPANCY)
    t = r.trips
    return {
        "key": f"{date_str}|{policy}",
        # (1440 x N) uint8 bike counts, row-major, base64 to keep JSON small
        "bikes": base64.b64encode(r.bikes.tobytes()).decode("ascii"),
        "reward": r.reward.round(2).tolist(),
        "fulfilled": r.fulfilled.tolist(),
        "lost": r.lost.tolist(),
        "active": r.active.tolist(),
        "blocked": r.blocked.tolist(),
        "pending": r.pending.tolist(),
        "trips": {
            "sm": t.start_min.tolist(), "em": t.end_min.tolist(),
            "s": t.start_idx.tolist(), "e": t.end_idx.tolist(),
            "casual": t.casual.tolist(), "ok": r.trip_started.astype(int).tolist(),
        },
        "n_requests": len(t),
    }


# ── Map figure ────────────────────────────────────────────────────────────────
# Every animation tick hands dcc.Graph a brand-new figure. To keep the user's
# zoom/pan, the clientside callback copies the live camera (center, zoom,
# bearing, pitch) into the outgoing layout, so Plotly.react has no view change
# to apply. `uirevision` alone is not enough: Plotly does not record wheel-zoom
# / drag as GUI edits for map subplots, so it snaps back to zoom=11.
# Basemap: OSM raster tiles drawn faded over white so stations/arcs stand out.
# Tune BASEMAP_OPACITY (0 = blank, 1 = full-strength OSM). The style `id` lets
# Plotly see it's the same style each tick instead of reloading it.
BASEMAP_OPACITY = 0.45
BASEMAP_STYLE = {
    "version": 8,
    "id": "faded-osm",
    "sources": {
        "osm": {
            "type": "raster",
            "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            "tileSize": 256,
            "maxzoom": 19,
            "attribution": "© OpenStreetMap contributors",
        },
    },
    "layers": [
        {"id": "background", "type": "background",
         "paint": {"background-color": "#ffffff"}},
        {"id": "osm", "type": "raster", "source": "osm",
         "paint": {"raster-opacity": BASEMAP_OPACITY, "raster-saturation": -0.3}},
    ],
}

MAP_LAYOUT = {
    "map": {"style": BASEMAP_STYLE, "center": TORONTO_CENTER, "zoom": 11},
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
    "showlegend": False,
    "uirevision": "keep-view",
}


def build_base_figure() -> dict:
    sdf = STATIONS_DF
    return {
        "data": [{
            "type": "scattermap",
            "lat": sdf["lat"].tolist(), "lon": sdf["lon"].tolist(),
            "mode": "markers",
            "marker": {"size": 4, "color": "#888888", "opacity": 0.4},
            "text": (sdf["name"] + "<br>capacity: "
                     + sdf["capacity"].astype(str) + " docks").tolist(),
            "hoverinfo": "text",
        }],
        "layout": MAP_LAYOUT,
    }


# ── Clientside animation callback ─────────────────────────────────────────────
# Rebuilds the figure in the browser each tick from the recorded simulation:
# station colour = fill level (bikes / docks), arcs = rides in progress (only
# requests the simulation actually served), plus the metric cards.
_ANIMATION_JS = """
function(current_min, sim, station_data, layout) {
    var nu = window.dash_clientside.no_update;
    if (!sim || !station_data || !layout ||
        current_min === null || current_min === undefined) {
        return Array(__N_OUTPUTS__).fill(nu);
    }
    var t = Math.max(0, Math.min(1439, current_min));

    // Decode the (1440 x N) bike matrix and cumulative sums once per simulation.
    if (!window._bsSim || window._bsSim.key !== sim.key) {
        var bin = atob(sim.bikes), u8 = new Uint8Array(bin.length);
        for (var b=0; b<bin.length; b++) u8[b] = bin.charCodeAt(b);
        var cum = {reward:[], fulfilled:[], lost:[], blocked:[]},
            acc = {reward:0, fulfilled:0, lost:0, blocked:0};
        for (var m=0; m<sim.reward.length; m++) {
            for (var key in acc) { acc[key] += sim[key][m]; cum[key].push(acc[key]); }
        }
        window._bsSim = {key: sim.key, bikes: u8, cum: cum};
    }
    var S = window._bsSim;

    var stLat=station_data.lat, stLon=station_data.lon;
    var stName=station_data.name, stCap=station_data.cap;
    var ns=stLat.length, row=t*ns;

    var color=new Array(ns), size=new Array(ns), text=new Array(ns);
    for (var k=0; k<ns; k++) {
        var nb=S.bikes[row+k], cap=stCap[k] || 1;
        color[k]=nb/cap;
        size[k]=5+10*Math.sqrt(cap/90);
        text[k]=stName[k]+'<br>bikes: '+nb+' / '+stCap[k]+' docks';
    }

    var tr=sim.trips, n=tr.sm.length;
    var mLat=[], mLon=[], cLat=[], cLon=[];
    for (var i=0; i<n; i++) {
        if (!tr.ok[i] || tr.sm[i] > t || tr.em[i] <= t) continue;
        var s=tr.s[i], e=tr.e[i];
        if (tr.casual[i]) {cLat.push(stLat[s],stLat[e],null); cLon.push(stLon[s],stLon[e],null);}
        else              {mLat.push(stLat[s],stLat[e],null); mLon.push(stLon[s],stLon[e],null);}
    }

    var hh=String(Math.floor(t/60)).padStart(2,'0');
    var mm=String(t%60).padStart(2,'0');
    var timeText=hh+':'+mm;

    function money(x) {
        return (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString('en-US',
            {minimumFractionDigits:2, maximumFractionDigits:2});
    }
    var requests = S.cum.fulfilled[t] + S.cum.lost[t];
    var pct = requests ? ' (' + (100*S.cum.fulfilled[t]/requests).toFixed(1) + '%)' : '';
    var metrics = [
        money(sim.reward[t]),
        money(S.cum.reward[t]),
        sim.fulfilled[t] + ' of ' + (sim.fulfilled[t] + sim.lost[t]),
        S.cum.fulfilled[t].toLocaleString() + pct,
        S.cum.lost[t].toLocaleString(),
        sim.active[t].toLocaleString(),
        sim.blocked[t].toLocaleString(),
        S.cum.blocked[t].toLocaleString(),
        sim.pending[t].toLocaleString()
    ];

    // Carry over the live camera so Plotly.react never moves the viewport.
    var viewLayout = Object.assign({}, layout);
    var gd = document.querySelector('#map .js-plotly-plot');
    var sub = gd && gd._fullLayout && gd._fullLayout.map && gd._fullLayout.map._subplot;
    if (sub && sub.map) {
        // Mid-gesture (wheel zoom easing, drag inertia): skip this redraw so
        // Plotly.react doesn't cut the camera animation short.
        if (sub.map.isMoving()) return [nu, timeText].concat(metrics);
        var c = sub.map.getCenter();
        viewLayout.map = Object.assign({}, layout.map, {
            center: {lat: c.lat, lon: c.lng},
            zoom: sub.map.getZoom(),
            bearing: sub.map.getBearing(),
            pitch: sub.map.getPitch()
        });
    }

    var figure = {
        data: [
            {type:'scattermap', mode:'lines', lat:mLat, lon:mLon,
             line:{color:'rgba(51,153,255,0.5)', width:1}, hoverinfo:'skip'},
            {type:'scattermap', mode:'lines', lat:cLat, lon:cLon,
             line:{color:'rgba(255,136,51,0.5)', width:1}, hoverinfo:'skip'},
            {type:'scattermap', mode:'markers', lat:stLat, lon:stLon,
             marker:{color:color, size:size, cmin:0, cmax:1, opacity:0.9,
                     colorscale:[[0,'#e8000b'],[0.15,'#ff9900'],[0.5,'#2ca02c'],[1,'#1f4fd1']]},
             text:text, hoverinfo:'text'}
        ],
        layout: viewLayout
    };

    return [figure, timeText].concat(metrics);
}
"""

# Metric cards under the map: (element id, label). Values come from the
# animation callback, for the minute shown on the slider.
METRICS = [
    ("m-reward", "Reward this minute"),
    ("m-cum-reward", "Cumulative reward"),
    ("m-fulfilled", "Requests fulfilled this minute"),
    ("m-cum-fulfilled", "Total requests fulfilled"),
    ("m-cum-lost", "Total requests lost"),
    ("m-active", "Active trips"),
    ("m-blocked", "Returns blocked this minute (station full)"),
    ("m-cum-blocked", "Total returns blocked"),
    ("m-pending", "Bikes pending (held 15 min)"),
]


def metric_card(card_id: str, label: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize":"12px","color":"#666"}),
            html.Div("–", id=card_id,
                     style={"fontSize":"20px","fontWeight":"bold","color":"#1a1a2e",
                            "fontVariantNumeric":"tabular-nums"}),
        ],
        style={"flex":"1 1 140px","background":"#fff","border":"1px solid #ddd",
               "borderRadius":"6px","padding":"8px 12px"},
    )


# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Bike Share Toronto")

app.layout = html.Div(
    [
        dcc.Store(id="sim-store", data=None),
        dcc.Store(id="station-store", data=_STATION_STORE),
        dcc.Store(id="map-layout", data=MAP_LAYOUT),
        dcc.Interval(id="interval", interval=500, disabled=True, n_intervals=0),

        # Header
        html.Div(
            [
                html.Span("Bike Share Toronto",
                          style={"fontWeight":"bold","fontSize":"18px","color":"white"}),
                html.Span("Date:", style={"color":"#ccc","marginLeft":"24px"}),
                dcc.DatePickerSingle(
                    id="date-picker",
                    min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                    date=DEFAULT_DATE, display_format="YYYY-MM-DD",
                ),
                html.Span("Policy:", style={"color":"#ccc","marginLeft":"16px"}),
                dcc.Dropdown(
                    id="policy-select",
                    options=[{"label": cls.label, "value": key}
                             for key, cls in POLICIES.items()],
                    value=DEFAULT_POLICY, clearable=False,
                    style={"width":"230px"},
                ),
                dcc.Loading(
                    html.Div(id="total-trips",
                             style={"color":"#aaffaa","marginLeft":"24px","fontSize":"14px"}),
                    type="dot", color="#aaffaa",
                ),
                html.Div(id="time-display", children="00:00",
                         style={"color":"#ffff88","marginLeft":"32px",
                                "fontSize":"15px","fontWeight":"bold"}),
            ],
            style={"display":"flex","alignItems":"center","gap":"8px",
                   "padding":"10px 16px","background":"#1a1a2e"},
        ),

        # Map container with legend overlay
        html.Div(
            [
                dcc.Graph(
                    id="map",
                    figure=build_base_figure(),
                    style={"height":"60vh"},
                    config={"scrollZoom": True, "displayModeBar": False},
                ),
                # Legend overlay
                html.Div(
                    [
                        html.Div("Station fill (bikes / docks)",
                                 style={"fontWeight":"bold","marginBottom":"4px",
                                        "fontSize":"12px"}),
                        html.Div([
                            html.Span("● empty  ",
                                      style={"color":"#e8000b","fontSize":"12px"}),
                            html.Span("● half  ",
                                      style={"color":"#2ca02c","fontSize":"12px"}),
                            html.Span("● full",
                                      style={"color":"#1f4fd1","fontSize":"12px"}),
                        ]),
                        html.Hr(style={"margin":"4px 0","borderColor":"#ccc"}),
                        html.Div(html.Span("— Member",
                                 style={"color":"#3399ff","fontSize":"12px"})),
                        html.Div(html.Span("— Casual",
                                 style={"color":"#ff8833","fontSize":"12px"})),
                    ],
                    style={
                        "position":"absolute","top":"10px","right":"10px",
                        "background":"rgba(255,255,255,0.88)",
                        "border":"1px solid #ccc","borderRadius":"4px",
                        "padding":"8px 10px","pointerEvents":"none","zIndex":1000,
                    },
                ),
            ],
            style={"position":"relative"},
        ),

        # Live simulation metrics (value at the minute shown on the slider)
        html.Div(
            [metric_card(cid, label) for cid, label in METRICS],
            style={"display":"flex","flexWrap":"wrap","gap":"8px",
                   "padding":"8px 20px","background":"#f4f4f4",
                   "borderTop":"1px solid #ddd"},
        ),

        # Controls
        html.Div(
            [
                html.Div(
                    [
                        html.Button("▶ Play", id="play-btn", n_clicks=0,
                                    style={"padding":"7px 18px","fontWeight":"bold",
                                           "cursor":"pointer","border":"1px solid #999",
                                           "borderRadius":"4px","background":"#fff",
                                           "fontSize":"14px"}),
                        html.Span("Speed:", style={"marginLeft":"16px","marginRight":"6px"}),
                        dcc.Dropdown(
                            id="speed-select",
                            options=[
                                {"label":"1×  (very slow)","value":1},
                                {"label":"5×  (slow)",     "value":5},
                                {"label":"10× (default)",  "value":10},
                                {"label":"30× (fast)",     "value":30},
                                {"label":"60× (very fast)","value":60},
                            ],
                            value=10, clearable=False,
                            style={"width":"155px","display":"inline-block"},
                        ),
                    ],
                    style={"display":"flex","alignItems":"center","marginBottom":"8px"},
                ),
                dcc.Slider(
                    id="time-slider", min=0, max=1439, step=1, value=0,
                    marks={i*60: f"{i:02d}:00" for i in range(0, 25, 2)},
                    tooltip={"placement":"bottom","always_visible":False},
                ),
            ],
            style={"padding":"10px 20px","background":"#f4f4f4",
                   "borderTop":"1px solid #ddd"},
        ),
    ],
    style={"fontFamily":"sans-serif"},
)


# ── Server callbacks ───────────────────────────────────────────────────────────

@app.callback(
    Output("total-trips", "children"),
    Output("sim-store", "data"),
    Input("date-picker", "date"),
    Input("policy-select", "value"),
)
def on_simulation_change(date_str, policy):
    if not date_str or not policy:
        return no_update, no_update
    sim = get_simulation(date_str, policy)
    return f"{sim['n_requests']:,} bike requests on {date_str}", sim


@app.callback(
    Output("interval", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, currently_disabled):
    now_playing = currently_disabled
    return (not now_playing), ("■ Stop" if now_playing else "▶ Play")


@app.callback(
    Output("time-slider", "value"),
    Input("interval", "n_intervals"),
    State("time-slider", "value"),
    State("speed-select", "value"),
    prevent_initial_call=True,
)
def advance_time(_, current_min, speed):
    return (current_min + speed) % 1440


# ── Clientside: animation — rebuilds the figure; uirevision keeps the view ────
app.clientside_callback(
    _ANIMATION_JS.replace("__N_OUTPUTS__", str(2 + len(METRICS))),
    Output("map", "figure"),
    Output("time-display", "children"),
    *[Output(cid, "children") for cid, _ in METRICS],
    Input("time-slider", "value"),
    Input("sim-store", "data"),
    State("station-store", "data"),
    State("map-layout", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    print(f"Simulating {DEFAULT_DATE} ...")
    get_simulation(DEFAULT_DATE, DEFAULT_POLICY)
    print("Starting Dash at http://localhost:8050")
    app.run(debug=False, port=8050)
