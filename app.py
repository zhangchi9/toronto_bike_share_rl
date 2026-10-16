"""
Bike Share Toronto — Animated Daily Trip Viewer
Usage:
    python fetch_stations.py     # once
    python preprocess.py         # once
    python app.py                # http://localhost:8050
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

from data_loader import load_stations, load_day

# ── Globals ───────────────────────────────────────────────────────────────────
STATIONS_DF = load_stations()
_day_cache: dict[str, pd.DataFrame] = {}

TORONTO_CENTER = {"lat": 43.655, "lon": -79.38}
MIN_DATE = "2026-01-01"
MAX_DATE = "2026-09-30"
DEFAULT_DATE = "2026-06-15"

# Station store built once: sent to browser on load, used by JS animation loop
_STATION_STORE = {
    "lat": STATIONS_DF["lat"].tolist(),
    "lon": STATIONS_DF["lon"].tolist(),
    "name": STATIONS_DF["name"].tolist(),
    "sid": STATIONS_DF["station_id"].tolist(),
    # JS lookup: station_id (string) → index in arrays above
    "idx": {str(int(sid)): i for i, sid in enumerate(STATIONS_DF["station_id"])},
}


def get_day_df(date_str: str) -> pd.DataFrame:
    if date_str not in _day_cache:
        df = load_day(date_str, STATIONS_DF)
        df = df.dropna(subset=["Start_Time", "End_Time", "start_lat", "start_lon", "end_lat", "end_lon"])
        day_start = pd.Timestamp(date_str)
        df["start_min"] = (
            (df["Start_Time"] - day_start).dt.total_seconds() / 60
        ).clip(0, 1439).astype(int)
        df["end_min"] = (
            (df["End_Time"] - day_start).dt.total_seconds() / 60
        ).clip(0, 1439).astype(int)
        _day_cache.clear()
        _day_cache[date_str] = df
    return _day_cache[date_str]


def serialize_trips(df: pd.DataFrame) -> dict:
    """Compact trip dict for the browser — all computation happens in JS."""
    return {
        "sm":   df["start_min"].tolist(),
        "em":   df["end_min"].tolist(),
        "slat": df["start_lat"].round(5).tolist(),
        "slon": df["start_lon"].round(5).tolist(),
        "elat": df["end_lat"].round(5).tolist(),
        "elon": df["end_lon"].round(5).tolist(),
        # 0 = Member, 1 = Casual
        "um":   (df["User_Type"] == "Casual").astype(int).tolist(),
        "ssid": df["Start_Station_Id"].tolist(),
        "esid": df["End_Station_Id"].tolist(),
    }


def build_base_figure() -> go.Figure:
    """
    Full figure sent once per date change.
    Uses go.Scattermapbox — all subsequent data updates go through
    Plotly.restyle() in JS so the map viewport is NEVER reset.
    """
    sdf = STATIONS_DF

    station_trace = go.Scattermapbox(
        lat=sdf["lat"].tolist(),
        lon=sdf["lon"].tolist(),
        mode="markers",
        name="Stations",
        marker=dict(
            size=[8] * len(sdf),
            color=[0.0] * len(sdf),
            colorscale=[[0, "#e8000b"], [0.5, "#f5d000"], [1, "#00c000"]],
            cmin=-1, cmax=1,
            showscale=True,
            colorbar=dict(
                title="bike flow",
                x=1.01, thickness=12, len=0.4,
                tickvals=[-1, 0, 1],
                ticktext=["depleted", "0", "replenished"],
            ),
            opacity=0.85,
        ),
        text=sdf["name"].tolist(),
        hoverinfo="text",
    )

    member_trace = go.Scattermapbox(
        lat=[], lon=[], mode="lines",
        name="Member (0)",
        line=dict(color="#3399ff", width=1),
        opacity=0.5, hoverinfo="skip",
    )
    casual_trace = go.Scattermapbox(
        lat=[], lon=[], mode="lines",
        name="Casual (0)",
        line=dict(color="#ff8833", width=1),
        opacity=0.5, hoverinfo="skip",
    )

    fig = go.Figure(data=[station_trace, member_trace, casual_trace])
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=TORONTO_CENTER,
            zoom=11,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
        ),
    )
    return fig


# ── Clientside animation callback (JavaScript) ────────────────────────────────
# Runs entirely in the browser on every slider tick.
#
# Why Plotly.react() instead of Plotly.restyle():
#   restyle() internally re-renders the Mapbox layer by reading
#   graphDiv.layout.mapbox.zoom/center.  Plotly does NOT update those fields
#   when the user pans/zooms interactively — so they always hold the original
#   figure values (zoom=11, Toronto centre), causing a visible viewport reset
#   on every frame.
#
#   Plotly.react() accepts a complete layout; by passing the user's CURRENT
#   zoom/center (captured via plotly_relayout events) we guarantee the map
#   always renders at exactly where the user is looking.
_ANIMATION_JS = """
async function(current_min, trip_data, station_data) {
    if (!trip_data || !station_data || current_min === null || current_min === undefined) {
        return window.dash_clientside.no_update;
    }

    var graphDiv = document.getElementById('map');
    if (!graphDiv || !graphDiv.data || graphDiv.data.length < 3) {
        return window.dash_clientside.no_update;
    }

    // ── One-time setup: track viewport via plotly_relayout ───────────────────
    // plotly_relayout fires whenever the user pans or zooms the map.
    // We cache the values so every animation frame can pass the current
    // viewport explicitly to Plotly.react(), preventing any reset.
    if (!graphDiv._vpTrackerReady) {
        graphDiv._vpTrackerReady = true;
        graphDiv._vp = {zoom: 11, center: {lat: 43.655, lon: -79.38}};
        graphDiv.on('plotly_relayout', function(ed) {
            if (!graphDiv._vp) graphDiv._vp = {};
            // Plotly may fire zoom/center as flat keys or nested under 'mapbox'
            if (ed['mapbox.zoom'] !== undefined)   graphDiv._vp.zoom   = ed['mapbox.zoom'];
            if (ed['mapbox.center'] !== undefined) graphDiv._vp.center = ed['mapbox.center'];
            if (ed.mapbox) {
                if (ed.mapbox.zoom   !== undefined) graphDiv._vp.zoom   = ed.mapbox.zoom;
                if (ed.mapbox.center !== undefined) graphDiv._vp.center = ed.mapbox.center;
            }
        });
    }

    // ── Computation ───────────────────────────────────────────────────────────
    var sm = trip_data.sm, em = trip_data.em;
    var slat = trip_data.slat, slon = trip_data.slon;
    var elat = trip_data.elat, elon = trip_data.elon;
    var um = trip_data.um;
    var ssid = trip_data.ssid, esid = trip_data.esid;
    var n = sm.length;

    var stLat = station_data.lat, stLon = station_data.lon;
    var stName = station_data.name;
    var sidIdx = station_data.idx;
    var ns = stLat.length;

    var dep = new Float32Array(ns);
    var arr = new Float32Array(ns);
    var mLats = [], mLons = [], cLats = [], cLons = [];
    var mCount = 0, cCount = 0;

    for (var i = 0; i < n; i++) {
        var s = sm[i], e = em[i];
        if (s <= current_min) { var di = sidIdx[String(ssid[i])]; if (di !== undefined) dep[di]++; }
        if (e <= current_min) { var ai = sidIdx[String(esid[i])]; if (ai !== undefined) arr[ai]++; }
        if (s <= current_min && e >= current_min) {
            if (um[i] === 0) {
                mLats.push(slat[i], elat[i], null);
                mLons.push(slon[i], elon[i], null);
                mCount++;
            } else {
                cLats.push(slat[i], elat[i], null);
                cLons.push(slon[i], elon[i], null);
                cCount++;
            }
        }
    }

    var maxAbs = 1, netFlow = new Float32Array(ns);
    for (var j = 0; j < ns; j++) {
        netFlow[j] = arr[j] - dep[j];
        if (Math.abs(netFlow[j]) > maxAbs) maxAbs = Math.abs(netFlow[j]);
    }

    var stColors = [], stSizes = [], stTexts = [];
    for (var k = 0; k < ns; k++) {
        var nf = netFlow[k] / maxAbs;
        stColors.push(nf);
        stSizes.push(Math.min(30, Math.max(6, 8 + Math.abs(netFlow[k]) * 0.5)));
        stTexts.push(stName[k] + '<br>Net flow: ' + Math.round(netFlow[k]));
    }

    // ── Viewport — use listener cache, fall back to Mapbox GL instance ────────
    var vp = graphDiv._vp || {zoom: 11, center: {lat: 43.655, lon: -79.38}};
    try {
        // Try every known internal path to the Mapbox GL instance
        var mb = (graphDiv._fullLayout.mapbox._subplot || {});
        var mapInst = mb.map || mb._map;
        if (mapInst && typeof mapInst.getZoom === 'function') {
            vp = {zoom: mapInst.getZoom(), center: (function(){
                var c = mapInst.getCenter(); return {lat: c.lat, lon: c.lng};
            })()};
            graphDiv._vp = vp;  // keep cache in sync
        }
    } catch(e) {}

    // ── Plotly.react with explicit viewport ───────────────────────────────────
    // Passing vp.zoom / vp.center in the layout is the authoritative way to
    // tell Plotly where to render the map.  Because we always supply the
    // user's CURRENT viewport, the map never jumps back to the original position.
    await Plotly.react(graphDiv,
        [
            {
                type: 'scattermapbox',
                lat: stLat, lon: stLon,
                mode: 'markers', name: 'Stations',
                marker: {
                    size: stSizes, color: stColors,
                    colorscale: [[0,'#e8000b'],[0.5,'#f5d000'],[1,'#00c000']],
                    cmin: -1, cmax: 1, showscale: true,
                    colorbar: {
                        title: 'bike flow', x: 1.01, thickness: 12, len: 0.4,
                        tickvals: [-1, 0, 1], ticktext: ['depleted', '0', 'replenished']
                    },
                    opacity: 0.85
                },
                text: stTexts, hoverinfo: 'text'
            },
            {
                type: 'scattermapbox',
                lat: mLats, lon: mLons,
                mode: 'lines', name: 'Member (' + mCount + ')',
                line: {color: '#3399ff', width: 1},
                opacity: 0.5, hoverinfo: 'skip'
            },
            {
                type: 'scattermapbox',
                lat: cLats, lon: cLons,
                mode: 'lines', name: 'Casual (' + cCount + ')',
                line: {color: '#ff8833', width: 1},
                opacity: 0.5, hoverinfo: 'skip'
            }
        ],
        {
            mapbox: {style: 'open-street-map', center: vp.center, zoom: vp.zoom},
            margin: {l: 0, r: 0, t: 0, b: 0},
            showlegend: true,
            legend: {x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1}
        }
    );

    var hh = String(Math.floor(current_min / 60)).padStart(2, '0');
    var mm = String(current_min % 60).padStart(2, '0');
    return hh + ':' + mm + '  |  Active: ' + (mCount + cCount);
}
"""


# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Bike Share Toronto")

app.layout = html.Div(
    [
        # Stores
        dcc.Store(id="loaded-date", data=None),
        dcc.Store(id="trip-store", data=None),
        dcc.Store(id="station-store", data=_STATION_STORE),
        dcc.Interval(id="interval", interval=500, disabled=True, n_intervals=0),

        # Header
        html.Div(
            [
                html.Span("Bike Share Toronto",
                          style={"fontWeight": "bold", "fontSize": "18px", "color": "white"}),
                html.Span("Date:", style={"color": "#ccc", "marginLeft": "24px"}),
                dcc.DatePickerSingle(
                    id="date-picker",
                    min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                    date=DEFAULT_DATE, display_format="YYYY-MM-DD",
                ),
                html.Div(id="total-trips",
                         style={"color": "#aaffaa", "marginLeft": "24px", "fontSize": "14px"}),
                # Time display updated by clientside JS — no server round-trip
                html.Div(id="time-display", children="00:00  |  Active: 0",
                         style={"color": "#ffff88", "marginLeft": "32px",
                                "fontSize": "15px", "fontWeight": "bold"}),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "padding": "10px 16px", "background": "#1a1a2e"},
        ),

        # Map — figure only replaced when date changes
        dcc.Graph(
            id="map",
            figure=build_base_figure(),
            style={"height": "67vh"},
            config={"scrollZoom": True},
        ),

        # Controls
        html.Div(
            [
                html.Div(
                    [
                        html.Button("▶ Play", id="play-btn", n_clicks=0,
                                    style={"padding": "7px 18px", "fontWeight": "bold",
                                           "cursor": "pointer", "border": "1px solid #999",
                                           "borderRadius": "4px", "background": "#fff",
                                           "fontSize": "14px"}),
                        html.Span("Speed:", style={"marginLeft": "16px", "marginRight": "6px"}),
                        dcc.Dropdown(
                            id="speed-select",
                            options=[
                                {"label": "1×  (very slow)", "value": 1},
                                {"label": "5×  (slow)", "value": 5},
                                {"label": "10× (default)", "value": 10},
                                {"label": "30× (fast)", "value": 30},
                                {"label": "60× (very fast)", "value": 60},
                            ],
                            value=10, clearable=False,
                            style={"width": "155px", "display": "inline-block"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                ),
                dcc.Slider(
                    id="time-slider", min=0, max=1439, step=1, value=0,
                    marks={i * 60: f"{i:02d}:00" for i in range(0, 25, 2)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"padding": "10px 20px", "background": "#f4f4f4", "borderTop": "1px solid #ddd"},
        ),
    ],
    style={"fontFamily": "sans-serif"},
)


# ── Server callbacks ───────────────────────────────────────────────────────────

@app.callback(
    Output("loaded-date", "data"),
    Output("total-trips", "children"),
    Output("trip-store", "data"),
    Input("date-picker", "date"),
)
def on_date_change(date_str):
    if not date_str:
        return no_update, no_update, no_update
    df = get_day_df(date_str)
    return date_str, f"{len(df):,} trips on {date_str}", serialize_trips(df)


@app.callback(
    Output("map", "figure"),
    Input("loaded-date", "data"),
)
def on_date_figure(date_str):
    """Replaces the full figure only when the date changes — resets view to Toronto."""
    if not date_str:
        return build_base_figure()
    get_day_df(date_str)  # ensure cached
    return build_base_figure()


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


# ── Clientside callback — runs in browser, calls Plotly.restyle directly ──────
app.clientside_callback(
    _ANIMATION_JS,
    Output("time-display", "children"),
    Input("time-slider", "value"),
    State("trip-store", "data"),
    State("station-store", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    print(f"Preloading {DEFAULT_DATE} ...")
    get_day_df(DEFAULT_DATE)
    print("Starting Dash at http://localhost:8050")
    app.run(debug=False, port=8050)
