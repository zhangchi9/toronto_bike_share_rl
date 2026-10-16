"""
Bike Share Toronto — Animated Daily Trip Viewer
Usage:
    python fetch_stations.py     # once
    python preprocess.py         # once
    python app.py                # http://localhost:8050
"""
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

_STATION_STORE = {
    "lat": STATIONS_DF["lat"].tolist(),
    "lon": STATIONS_DF["lon"].tolist(),
    "name": STATIONS_DF["name"].tolist(),
    "sid": STATIONS_DF["station_id"].tolist(),
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
    return {
        "sm":   df["start_min"].tolist(),
        "em":   df["end_min"].tolist(),
        "slat": df["start_lat"].round(5).tolist(),
        "slon": df["start_lon"].round(5).tolist(),
        "elat": df["end_lat"].round(5).tolist(),
        "elon": df["end_lon"].round(5).tolist(),
        "um":   (df["User_Type"] == "Casual").astype(int).tolist(),
        "ssid": df["Start_Station_Id"].tolist(),
        "esid": df["End_Station_Id"].tolist(),
    }


def build_base_figure() -> go.Figure:
    """
    Minimal figure — just enough to create the Mapbox GL map.
    All animated data (station colors, arcs) is rendered via raw Mapbox GL
    API in the clientside JS callback, bypassing Plotly entirely so the
    map viewport is NEVER reset.
    """
    sdf = STATIONS_DF

    # Station trace: faint dots so initial load shows station positions.
    # Once animation starts, the Mapbox GL 'bs-station-layer' overlays these
    # with flow-colored circles.
    station_trace = go.Scattermapbox(
        lat=sdf["lat"].tolist(),
        lon=sdf["lon"].tolist(),
        mode="markers",
        name="Stations",
        marker=dict(size=4, color="#888888", opacity=0.4),
        text=sdf["name"].tolist(),
        hoverinfo="text",
    )

    fig = go.Figure(data=[station_trace])
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=TORONTO_CENTER, zoom=11),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


# ── Clientside animation callback ─────────────────────────────────────────────
#
# Architecture: completely bypass Plotly for all animation updates.
# Instead, use the raw Mapbox GL JS API:
#   map.addSource() / map.addLayer()  — one-time setup on first tick
#   map.getSource(id).setData(geojson) — every subsequent tick
#
# map.getSource().setData() ONLY updates the GeoJSON data fed to the GPU
# renderer.  It never touches the camera / viewport, so zoom/pan is
# permanently preserved no matter how fast the animation runs.
_ANIMATION_JS = """
function(current_min, trip_data, station_data) {
    if (!trip_data || !station_data || current_min === null || current_min === undefined) {
        return window.dash_clientside.no_update;
    }

    var graphDiv = document.getElementById('map');
    if (!graphDiv || !graphDiv._fullLayout) {
        return window.dash_clientside.no_update;
    }

    // ── Find the underlying Mapbox GL / MapLibre GL map instance ─────────────
    // Plotly stores it somewhere inside _fullLayout; we try every known path
    // plus a broad scan so this works across Plotly versions.
    function findMap() {
        var fl = graphDiv._fullLayout;
        var directPaths = [
            function(){ return fl.mapbox._subplot.map;     },
            function(){ return fl.mapbox._subplot._map;    },
            function(){ return fl.map._subplot.map;        },
            function(){ return fl.map._subplot._map;       },
            function(){ return fl.mapbox._subplot.maplibre;},
        ];
        for (var i = 0; i < directPaths.length; i++) {
            try { var m = directPaths[i](); if (m && m.getZoom) return m; } catch(e) {}
        }
        // Broad scan: walk all _fullLayout keys looking for an object with getZoom
        try {
            var keys = Object.keys(fl);
            for (var ki = 0; ki < keys.length; ki++) {
                var obj = fl[keys[ki]];
                if (!obj || typeof obj !== 'object') continue;
                var sub = obj._subplot;
                if (!sub || typeof sub !== 'object') continue;
                var skeys = Object.keys(sub);
                for (var si = 0; si < skeys.length; si++) {
                    var v = sub[skeys[si]];
                    if (v && typeof v.getZoom === 'function') return v;
                }
            }
        } catch(e) {}
        return null;
    }

    var map = findMap();
    if (!map) return window.dash_clientside.no_update;

    // ── One-time layer setup (keyed to the Mapbox GL instance) ───────────────
    // When the date changes, Plotly rebuilds the map → new instance, _bsSetup
    // is absent → we re-add our layers automatically.
    if (!map._bsSetup) {
        // Wait until the map style is fully loaded before adding layers
        if (!map.isStyleLoaded()) {
            map.once('style.load', function() { map._bsSetup = false; });
            return window.dash_clientside.no_update;
        }

        map._bsSetup = true;

        // Clean up stale layers/sources from a previous session (hot-reload etc.)
        ['bs-casual-layer','bs-member-layer','bs-station-layer'].forEach(function(id){
            try { if (map.getLayer(id)) map.removeLayer(id); } catch(e) {}
        });
        ['bs-casual-arcs','bs-member-arcs','bs-stations'].forEach(function(id){
            try { if (map.getSource(id)) map.removeSource(id); } catch(e) {}
        });

        // Station circles — color = net flow, radius = |flow| magnitude
        map.addSource('bs-stations', {type:'geojson', data:{type:'FeatureCollection',features:[]}});
        map.addLayer({
            id: 'bs-station-layer', type: 'circle', source: 'bs-stations',
            paint: {
                'circle-color': ['interpolate',['linear'],['get','flow'],
                    -1,'#e8000b', 0,'#f5d000', 1,'#00c000'],
                'circle-radius': ['interpolate',['linear'],['get','r'], 0,6, 1,18],
                'circle-opacity': 0.9,
                'circle-stroke-width': 1,
                'circle-stroke-color': 'rgba(0,0,0,0.25)'
            }
        });

        // Member trip arcs (blue)
        map.addSource('bs-member-arcs', {type:'geojson', data:{type:'FeatureCollection',features:[]}});
        map.addLayer({
            id: 'bs-member-layer', type: 'line', source: 'bs-member-arcs',
            paint: {'line-color':'#3399ff','line-width':1,'line-opacity':0.5}
        });

        // Casual trip arcs (orange)
        map.addSource('bs-casual-arcs', {type:'geojson', data:{type:'FeatureCollection',features:[]}});
        map.addLayer({
            id: 'bs-casual-layer', type: 'line', source: 'bs-casual-arcs',
            paint: {'line-color':'#ff8833','line-width':1,'line-opacity':0.5}
        });
    }

    // ── Compute per-station flow and active arcs ──────────────────────────────
    var sm=trip_data.sm, em=trip_data.em;
    var slat=trip_data.slat, slon=trip_data.slon;
    var elat=trip_data.elat, elon=trip_data.elon;
    var um=trip_data.um, ssid=trip_data.ssid, esid=trip_data.esid;
    var n=sm.length;

    var stLat=station_data.lat, stLon=station_data.lon;
    var stName=station_data.name, sidIdx=station_data.idx;
    var ns=stLat.length;

    var dep=new Float32Array(ns), arr=new Float32Array(ns);
    var mFeatures=[], cFeatures=[];
    var mCount=0, cCount=0;

    for (var i=0; i<n; i++) {
        var s=sm[i], e=em[i];
        if (s<=current_min) { var di=sidIdx[String(ssid[i])]; if(di!==undefined) dep[di]++; }
        if (e<=current_min) { var ai=sidIdx[String(esid[i])]; if(ai!==undefined) arr[ai]++; }
        if (s<=current_min && e>=current_min) {
            var feat={type:'Feature',
                geometry:{type:'LineString',
                    coordinates:[[slon[i],slat[i]],[elon[i],elat[i]]]}};
            if (um[i]===0) { mFeatures.push(feat); mCount++; }
            else            { cFeatures.push(feat); cCount++; }
        }
    }

    var maxAbs=1, netFlow=new Float32Array(ns);
    for (var j=0; j<ns; j++) {
        netFlow[j]=arr[j]-dep[j];
        if (Math.abs(netFlow[j])>maxAbs) maxAbs=Math.abs(netFlow[j]);
    }

    // ── Build GeoJSON for stations ────────────────────────────────────────────
    var stFeatures=[];
    for (var k=0; k<ns; k++) {
        var nf=netFlow[k]/maxAbs;                  // -1 … 1  (colour)
        var r=Math.abs(nf);                         //  0 … 1  (radius)
        stFeatures.push({type:'Feature',
            geometry:{type:'Point',coordinates:[stLon[k],stLat[k]]},
            properties:{name:stName[k], flow:nf, r:r, net:Math.round(netFlow[k])}
        });
    }

    // ── Push new data to Mapbox GL — NO Plotly calls, viewport untouched ──────
    map.getSource('bs-stations').setData(
        {type:'FeatureCollection', features:stFeatures});
    map.getSource('bs-member-arcs').setData(
        {type:'FeatureCollection', features:mFeatures});
    map.getSource('bs-casual-arcs').setData(
        {type:'FeatureCollection', features:cFeatures});

    var hh=String(Math.floor(current_min/60)).padStart(2,'0');
    var mm2=String(current_min%60).padStart(2,'0');
    return hh+':'+mm2+'  |  Active: '+(mCount+cCount)
           +'  (M:'+mCount+' C:'+cCount+')';
}
"""


# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Bike Share Toronto")

app.layout = html.Div(
    [
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
                html.Div(id="time-display", children="00:00",
                         style={"color": "#ffff88", "marginLeft": "32px",
                                "fontSize": "15px", "fontWeight": "bold"}),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "padding": "10px 16px", "background": "#1a1a2e"},
        ),

        # Map container with legend overlay
        html.Div(
            [
                dcc.Graph(
                    id="map",
                    figure=build_base_figure(),
                    style={"height": "67vh"},
                    config={"scrollZoom": True},
                ),
                # Colour-coded legend (HTML overlay — no Plotly colorbar needed)
                html.Div(
                    [
                        html.Div("Station flow", style={"fontWeight": "bold",
                                                        "marginBottom": "4px",
                                                        "fontSize": "12px"}),
                        html.Div([
                            html.Span("▲ bikes in  ",
                                      style={"color": "#00c000", "fontSize": "12px"}),
                            html.Span("● neutral  ",
                                      style={"color": "#f5d000", "fontSize": "12px"}),
                            html.Span("▼ bikes out",
                                      style={"color": "#e8000b", "fontSize": "12px"}),
                        ]),
                        html.Hr(style={"margin": "4px 0", "borderColor": "#ccc"}),
                        html.Div([
                            html.Span("— Member",
                                      style={"color": "#3399ff", "fontSize": "12px"}),
                        ]),
                        html.Div([
                            html.Span("— Casual",
                                      style={"color": "#ff8833", "fontSize": "12px"}),
                        ]),
                    ],
                    style={
                        "position": "absolute", "top": "10px", "right": "10px",
                        "background": "rgba(255,255,255,0.88)",
                        "border": "1px solid #ccc", "borderRadius": "4px",
                        "padding": "8px 10px", "pointerEvents": "none",
                        "zIndex": 1000,
                    },
                ),
            ],
            style={"position": "relative"},
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
                                {"label": "5×  (slow)",      "value": 5},
                                {"label": "10× (default)",   "value": 10},
                                {"label": "30× (fast)",      "value": 30},
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
            style={"padding": "10px 20px", "background": "#f4f4f4",
                   "borderTop": "1px solid #ddd"},
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
    if not date_str:
        return build_base_figure()
    get_day_df(date_str)
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


# ── Clientside callback — pure Mapbox GL, zero Plotly calls during animation ──
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
