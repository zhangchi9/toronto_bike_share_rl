"""
Bike Share Toronto — Animated Daily Trip Viewer
Usage:
    python fetch_stations.py     # once
    python preprocess.py         # once
    python app.py                # http://localhost:8050
"""
import pandas as pd
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
    "cap":  STATIONS_DF["capacity"].tolist(),
    "sid": STATIONS_DF["station_id"].tolist(),
    "idx": {str(int(sid)): i for i, sid in enumerate(STATIONS_DF["station_id"])},
}


def get_day_df(date_str: str) -> pd.DataFrame:
    if date_str not in _day_cache:
        df = load_day(date_str, STATIONS_DF)
        df = df.dropna(subset=["Start_Time", "End_Time",
                                "start_lat", "start_lon", "end_lat", "end_lon"])
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
# Rebuilds the figure in the browser each tick. Arcs are drawn as one line
# trace per user type, with null separators between segments.
_ANIMATION_JS = """
function(current_min, trip_data, station_data, layout) {
    var nu = window.dash_clientside.no_update;
    if (!trip_data || !station_data || !layout ||
        current_min === null || current_min === undefined) {
        return [nu, nu];
    }

    var sm=trip_data.sm, em=trip_data.em;
    var slat=trip_data.slat, slon=trip_data.slon;
    var elat=trip_data.elat, elon=trip_data.elon;
    var um=trip_data.um, ssid=trip_data.ssid, esid=trip_data.esid;
    var n=sm.length;

    var stLat=station_data.lat, stLon=station_data.lon;
    var stName=station_data.name, stCap=station_data.cap, sidIdx=station_data.idx;
    var ns=stLat.length;

    var dep=new Float32Array(ns), arr=new Float32Array(ns);
    var mLat=[], mLon=[], cLat=[], cLon=[];
    var mCount=0, cCount=0;

    for (var i=0; i<n; i++) {
        var s=sm[i], e=em[i];
        if (s<=current_min){var di=sidIdx[String(ssid[i])];if(di!==undefined)dep[di]++;}
        if (e<=current_min){var ai=sidIdx[String(esid[i])];if(ai!==undefined)arr[ai]++;}
        if (s<=current_min && e>=current_min) {
            if (um[i]===0){mLat.push(slat[i],elat[i],null);mLon.push(slon[i],elon[i],null);mCount++;}
            else          {cLat.push(slat[i],elat[i],null);cLon.push(slon[i],elon[i],null);cCount++;}
        }
    }

    var netFlow=new Float32Array(ns), maxAbs=1;
    for (var j=0; j<ns; j++) {
        netFlow[j]=arr[j]-dep[j];
        if (Math.abs(netFlow[j])>maxAbs) maxAbs=Math.abs(netFlow[j]);
    }

    var color=new Array(ns), size=new Array(ns), text=new Array(ns);
    for (var k=0; k<ns; k++) {
        var nf=netFlow[k]/maxAbs;
        color[k]=nf;
        size[k]=6+12*Math.abs(nf);
        text[k]=stName[k]+'<br>capacity: '+stCap[k]+' docks'
                +'<br>net: '+Math.round(netFlow[k]);
    }

    var hh=String(Math.floor(current_min/60)).padStart(2,'0');
    var mm=String(current_min%60).padStart(2,'0');
    var timeText=hh+':'+mm+'  |  Active:'+(mCount+cCount)
                 +' (M:'+mCount+' C:'+cCount+')';

    // Carry over the live camera so Plotly.react never moves the viewport.
    var viewLayout = Object.assign({}, layout);
    var gd = document.querySelector('#map .js-plotly-plot');
    var sub = gd && gd._fullLayout && gd._fullLayout.map && gd._fullLayout.map._subplot;
    if (sub && sub.map) {
        // Mid-gesture (wheel zoom easing, drag inertia): skip this redraw so
        // Plotly.react doesn't cut the camera animation short.
        if (sub.map.isMoving()) return [nu, timeText];
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
             marker:{color:color, size:size, cmin:-1, cmax:1, opacity:0.9,
                     colorscale:[[0,'#e8000b'],[0.5,'#f5d000'],[1,'#00c000']]},
             text:text, hoverinfo:'text'}
        ],
        layout: viewLayout
    };

    return [figure, timeText];
}
"""


# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Bike Share Toronto")

app.layout = html.Div(
    [
        dcc.Store(id="loaded-date", data=None),
        dcc.Store(id="trip-store",  data=None),
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
                html.Div(id="total-trips",
                         style={"color":"#aaffaa","marginLeft":"24px","fontSize":"14px"}),
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
                    style={"height":"67vh"},
                    config={"scrollZoom": True, "displayModeBar": False},
                ),
                # Legend overlay
                html.Div(
                    [
                        html.Div("Station flow",
                                 style={"fontWeight":"bold","marginBottom":"4px",
                                        "fontSize":"12px"}),
                        html.Div([
                            html.Span("▲ bikes in  ",
                                      style={"color":"#00c000","fontSize":"12px"}),
                            html.Span("● neutral  ",
                                      style={"color":"#aa8800","fontSize":"12px"}),
                            html.Span("▼ bikes out",
                                      style={"color":"#e8000b","fontSize":"12px"}),
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
    _ANIMATION_JS,
    Output("map", "figure"),
    Output("time-display", "children"),
    Input("time-slider", "value"),
    Input("trip-store", "data"),
    State("station-store", "data"),
    State("map-layout", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    print(f"Preloading {DEFAULT_DATE} ...")
    get_day_df(DEFAULT_DATE)
    print("Starting Dash at http://localhost:8050")
    app.run(debug=False, port=8050)
