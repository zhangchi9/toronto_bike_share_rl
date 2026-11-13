"""
Bike Share Toronto — simulation replay viewer
Replays one continuous simulated run per rebalancing policy, Aug 1 - Sep 30, 2026
(stations filled once at 00:00 on Aug 1; bikes carry over between days). Pick any
date range in that period and a policy: station fill levels, active rides, trucks
and live metrics are animated minute by minute through the range.
Usage:
    python fetch_stations.py         # once
    python preprocess.py             # once
    python build_road_distances.py   # once
    python -m dashboard_data         # once, and after retraining (about 2 min)
    python app.py                    # http://localhost:8050
"""
import threading

import pandas as pd
from dash import Dash, ctx, dcc, html, Input, Output, State, no_update

import dashboard_data as dd           # also adds the trained RL models to POLICIES
from bikeshare_env.env import MINUTES_PER_DAY
from bikeshare_env.policies import POLICIES
from data_loader import load_stations

# ── Globals ───────────────────────────────────────────────────────────────────
STATIONS_DF = load_stations()

TORONTO_CENTER = {"lat": 43.655, "lon": -79.38}
MIN_DATE, MAX_DATE = dd.REPLAY_START, dd.REPLAY_END    # other dates are greyed out
DEFAULT_RANGE = ("2026-08-01", "2026-08-07")
DEFAULT_POLICY = "no_rebalancing"

# RL zones (rl/zones.py) drawn as an optional overlay: outline + centre label.
try:
    from rl.wrapper import static_data
    from rl.zones import zone_outlines
    _ZONE_STORE = zone_outlines(static_data()["zones"], STATIONS_DF["lat"],
                                STATIONS_DF["lon"], STATIONS_DF["capacity"])
except Exception as e:   # noqa: BLE001 - overlay is optional
    print(f"Zone overlay disabled: {e}")
    _ZONE_STORE = []

_STATION_STORE = {
    "lat": STATIONS_DF["lat"].tolist(),
    "lon": STATIONS_DF["lon"].tolist(),
    "name": STATIONS_DF["name"].tolist(),
    "cap":  STATIONS_DF["capacity"].tolist(),
}


def _day(date_str: str) -> int:
    """Replay day index (0 = Aug 1) of a date picker value."""
    return (pd.Timestamp(date_str).normalize() - pd.Timestamp(dd.REPLAY_START)).days


def _label(d: int, weekday: bool = False) -> str:
    date = dd.day_date(d)
    return (f"{date:%a} " if weekday else "") + f"{date:%b} {date.day}"


# ── Timeline marks: the date at each midnight, then every 3 hours ─────────────
_HOUR_MARK = {"color": "#777", "fontSize": "11px"}
_DATE_MARK = {"fontWeight": "bold", "color": "#1a1a2e"}
PX_PER_DAY = 300          # timeline width per day, so 3-hour labels never overlap


def _hour_label(h: int) -> str:
    return "12pm" if h == 12 else f"{h % 12}{'am' if h < 12 else 'pm'}"


def slider_marks(n_days: int, d0: int) -> dict:
    marks = {}
    for i in range(n_days):
        start = i * MINUTES_PER_DAY
        marks[start] = {"label": _label(d0 + i), "style": _DATE_MARK}
        for h in range(3, 24, 3):
            marks[start + h * 60] = {"label": _hour_label(h), "style": _HOUR_MARK}
    return marks


def slider_width(n_days: int) -> dict:
    """At least the page width; longer ranges scroll sideways."""
    # Side padding keeps the first date label and the handle at minute 0 unclipped.
    return {"width": f"max(100%, {n_days * PX_PER_DAY}px)", "padding": "0 28px",
            "boxSizing": "border-box"}


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

# Truck icons come from a sprite sheet in assets/ (make_truck_icons.py). MapLibre
# needs an absolute sprite URL, so the layout is built per page load.
TRUCK_SPRITE_PATH = "assets/truck_sprite/sprite"


def map_layout(base_url: str) -> dict:
    # Plotly's icon layers declare a text font, and MapLibre rejects such a layer
    # unless the style has a glyphs URL (it's never fetched: no text is drawn).
    style = dict(BASEMAP_STYLE, id="faded-osm-trucks", sprite=base_url + TRUCK_SPRITE_PATH,
                 glyphs="https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf")
    return {
        "map": {"style": style, "center": TORONTO_CENTER, "zoom": 11},
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": False,
        "uirevision": "keep-view",
    }


def build_base_figure(layout: dict) -> dict:
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
        "layout": layout,
    }


# ── Clientside animation callback ─────────────────────────────────────────────
# Rebuilds the figure in the browser each tick from the recorded simulation:
# station colour = fill level (bikes / docks), arcs = rides in progress (only
# requests the simulation actually served), plus the metric cards.
_ANIMATION_JS = """
function(current_min, show_zones, sim, station_data, layout, zones) {
    var nu = window.dash_clientside.no_update;
    if (!sim || !station_data || !layout ||
        current_min === null || current_min === undefined) {
        return Array(__N_OUTPUTS__).fill(nu);
    }
    // The slider counts minutes from 00:00 on the first picked day; `sim` holds one
    // day of the replay (minutes relative to its 00:00). While the next day loads, wait.
    window._bsRangeStart = sim.start;    // for the slider tooltip (assets/slider_time.js)
    var t = current_min - sim.day*1440;
    if (t < 0 || t > 1439) return Array(__N_OUTPUTS__).fill(nu);

    // Long ranges scroll sideways: keep the play head in view.
    var scroller = document.getElementById('slider-scroll');
    if (scroller && scroller.scrollWidth > scroller.clientWidth + 1 && sim.n_days) {
        var pad = 12 + 28, track = scroller.scrollWidth - 2*pad;   // scroller + inner padding
        var x = pad + track * current_min / Math.max(1, sim.n_days*1440 - 1);
        var left = scroller.scrollLeft, w = scroller.clientWidth;
        if (x < left + 60 || x > left + w - 60) scroller.scrollLeft = Math.max(0, x - w*0.25);
    }

    // Decode the (1440 x N) bike matrix and cumulative sums once per simulation.
    if (!window._bsSim || window._bsSim.key !== sim.key) {
        var bin = atob(sim.bikes), u8 = new Uint8Array(bin.length);
        for (var b=0; b<bin.length; b++) u8[b] = bin.charCodeAt(b);
        var cum = {reward:[], fulfilled:[], lost:[], blocked:[]},
            acc = Object.assign({}, sim.cum0);   // running totals since the range start
        for (var m=0; m<sim.reward.length; m++) {
            for (var key in acc) { acc[key] += sim[key][m]; cum[key].push(acc[key]); }
        }
        // Per truck: its trips in time order, with cumulative distance along the route.
        var legsBy = [];
        if (sim.trucks) {
            for (var q=0; q<sim.trucks.n; q++) legsBy.push([]);
            sim.trucks.legs.forEach(function(L) {
                var d=[0];
                for (var p=1; p<L.lat.length; p++) {
                    var dy=L.lat[p]-L.lat[p-1], dx=(L.lon[p]-L.lon[p-1])*0.72;
                    d.push(d[p-1]+Math.sqrt(dx*dx+dy*dy));
                }
                L.cum=d; legsBy[L.k].push(L);
            });
        }
        window._bsSim = {key: sim.key, bikes: u8, cum: cum, legs: legsBy};
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
    var timeText=sim.date+'  '+hh+':'+mm;

    // ── Trucks: position along the road route of the trip in progress ──────────
    function clock(m) {
        return String(Math.floor(m/60)%24).padStart(2,'0')+':'+String(m%60).padStart(2,'0');
    }
    var truckTraces=[], truckCard='No trucks', movedCard='–';
    var TRUCK_COL=['#7b1fa2','#00838f','#ad1457','#2e7d32'];   // route line per truck
    var ICON={drive_full:'full', drive_empty:'empty', handling:'handling', idle:'idle'};
    if (sim.trucks) {
        var TK=sim.trucks, icons={lat:[],lon:[],symbol:[],text:[]};
        var cardParts=[];
        S.facing = S.facing || [];
        for (var k=0; k<TK.n; k++) {
            var st=TK.status[k][t], load=TK.load[k][t], at=TK.station[k][t];
            var plat=stLat[at], plon=stLon[at], desc, state, leg=null;
            if (st===1) {
                var Ls=S.legs[k];
                for (var j=Ls.length-1; j>=0; j--) {
                    if (Ls[j].depart<=t && t<Ls[j].arrive) { leg=Ls[j]; break; }
                }
            }
            if (leg) {
                // Fraction of the drive done by the end of minute t.
                var f=Math.min(1, (t+1-leg.depart)/Math.max(1, leg.arrive-leg.depart));
                var target=f*leg.cum[leg.cum.length-1], p=1;
                while (p<leg.cum.length-1 && leg.cum[p]<target) p++;
                var seg=leg.cum[p]-leg.cum[p-1], w=seg>0 ? (target-leg.cum[p-1])/seg : 1;
                plat=leg.lat[p-1]+w*(leg.lat[p]-leg.lat[p-1]);
                plon=leg.lon[p-1]+w*(leg.lon[p]-leg.lon[p-1]);
                // Face the direction of travel on the current road segment.
                var dx=leg.lon[p]-leg.lon[p-1];
                if (Math.abs(dx)>1e-6) S.facing[k] = dx>0 ? 'r' : 'l';
                var col=TRUCK_COL[k%TRUCK_COL.length];
                truckTraces.push({type:'scattermap', mode:'lines', hoverinfo:'skip',
                    lat:leg.lat.slice(0,p).concat([plat]), lon:leg.lon.slice(0,p).concat([plon]),
                    line:{color:col, width:3}, opacity:0.35});
                truckTraces.push({type:'scattermap', mode:'lines', hoverinfo:'skip',
                    lat:[plat].concat(leg.lat.slice(p)), lon:[plon].concat(leg.lon.slice(p)),
                    line:{color:col, width:5}, opacity:0.9});
                state = leg.load>0 ? 'drive_full' : 'drive_empty';
                desc = 'driving '+(leg.load>0 ? leg.load+' bikes' : 'empty')+' to '
                       +stName[leg.to]+', arrives '+clock(leg.arrive);
                cardParts.push('T'+(k+1)+': \u2192 '+stName[leg.to]
                               +(leg.load>0 ? ' ('+leg.load+' bikes)' : ' (empty)'));
            } else if (st===2) {
                state='handling';
                desc='loading / unloading at '+stName[at]+' ('+load+' bikes on board)';
                cardParts.push('T'+(k+1)+': loading/unloading at '+stName[at]);
            } else {
                state='idle';
                desc='idle at '+stName[at];
                cardParts.push('T'+(k+1)+': idle');
            }
            icons.lat.push(plat); icons.lon.push(plon);
            icons.symbol.push('truck-'+ICON[state]+'-'+(S.facing[k] || 'r'));
            icons.text.push('<b>Truck '+(k+1)+'</b><br>'+desc);
        }
        truckTraces.push({type:'scattermap', mode:'markers', hoverinfo:'text',
            lat:icons.lat, lon:icons.lon, text:icons.text,
            marker:{symbol:icons.symbol, size:14, allowoverlap:true}});
        truckCard=cardParts.join(' \u00b7 ');
        movedCard=TK.moved[t].toLocaleString();
    }

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
        sim.pending[t].toLocaleString(),
        truckCard,
        movedCard
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

    // ── Zone overlay (outlines under everything, numbers on top) ──────────────
    var zoneTraces=[], zoneLabels=[];
    if (zones && zones.length && show_zones && show_zones.indexOf('on')>=0) {
        var ZC=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2',
                '#7f7f7f','#bcbd22','#17becf','#393b79','#637939','#8c6d31','#843c39',
                '#7b4173','#3182bd','#e6550d','#31a354','#756bb1','#636363',
                '#111111'];   // 21st: the island zone (round 5 zones)
        zones.forEach(function(z, i) {
            var c=ZC[i%ZC.length];
            var r=parseInt(c.slice(1,3),16), g=parseInt(c.slice(3,5),16), b=parseInt(c.slice(5,7),16);
            zoneTraces.push({type:'scattermap', mode:'lines', fill:'toself',
                fillcolor:'rgba('+r+','+g+','+b+',0.08)', lat:z.lat, lon:z.lon,
                line:{color:c, width:2}, hoverinfo:'text',
                text:'Zone '+z.zone+': '+z.stations+' stations, '+z.docks.toLocaleString()+' docks'});
        });
        zoneLabels.push({type:'scattermap', mode:'text', hoverinfo:'skip',
            lat:zones.map(function(z){return z.clat;}), lon:zones.map(function(z){return z.clon;}),
            text:zones.map(function(z){return 'Z'+z.zone;}),
            // Fonts come from the style's glyphs server (demotiles has Noto Sans Bold).
            textfont:{size:15, color:'#111111', family:'Noto Sans Bold'}});
    }

    var figure = {
        data: zoneTraces.concat([
            {type:'scattermap', mode:'lines', lat:mLat, lon:mLon,
             line:{color:'rgba(51,153,255,0.5)', width:1}, hoverinfo:'skip'},
            {type:'scattermap', mode:'lines', lat:cLat, lon:cLon,
             line:{color:'rgba(255,136,51,0.5)', width:1}, hoverinfo:'skip'},
            {type:'scattermap', mode:'markers', lat:stLat, lon:stLon,
             marker:{color:color, size:size, cmin:0, cmax:1, opacity:0.9,
                     colorscale:[[0,'#e8000b'],[0.15,'#ff9900'],[0.5,'#2ca02c'],[1,'#1f4fd1']]},
             text:text, hoverinfo:'text'}
        ]).concat(zoneLabels).concat(truckTraces),
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
    ("m-trucks", "Trucks"),
    ("m-truck-moved", "Bikes moved by trucks"),
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

def serve_layout():
    from flask import has_request_context, request
    # Dash also calls this once at startup (no request) to validate the layout.
    layout = map_layout(request.url_root if has_request_context() else "http://localhost:8050/")
    return html.Div(
    [
        dcc.Store(id="sim-store", data=None),      # the replay day being shown
        dcc.Store(id="range-store", data=None),    # {policy, d0, n_days} of the picked range
        dcc.Store(id="station-store", data=_STATION_STORE),
        dcc.Store(id="zone-store", data=_ZONE_STORE),
        dcc.Store(id="map-layout", data=layout),
        dcc.Interval(id="interval", interval=500, disabled=True, n_intervals=0),

        # Header
        html.Div(
            [
                html.Span("Bike Share Toronto",
                          style={"fontWeight":"bold","fontSize":"18px","color":"white"}),
                html.Span("Dates:", style={"color":"#ccc","marginLeft":"24px"}),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                    start_date=DEFAULT_RANGE[0], end_date=DEFAULT_RANGE[1],
                    initial_visible_month=MIN_DATE, minimum_nights=0,
                    display_format="YYYY-MM-DD",
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
                dcc.Checklist(
                    id="zone-toggle",
                    options=[{"label": " Show zones", "value": "on"}],
                    value=["on"],
                    style={"marginLeft":"16px","fontSize":"14px"},
                    labelStyle={"color":"#eeeeee","cursor":"pointer"},
                ),
                html.Div(id="time-display", children="",
                         style={"color":"#ffff88","marginLeft":"32px",
                                "fontSize":"15px","fontWeight":"bold",
                                "fontVariantNumeric":"tabular-nums"}),
                html.Div(
                    [html.Span("Jump to day:", style={"color":"#ccc","marginRight":"6px"}),
                     dcc.Dropdown(id="day-select", options=[], value=0, clearable=False,
                                  searchable=False, style={"width":"140px"})],
                    id="day-select-wrap",
                    style={"display":"none","alignItems":"center","marginLeft":"12px"},
                ),
            ],
            # Above the map legend, so the calendar and dropdown menus aren't hidden behind it.
            style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"wrap",
                   "padding":"10px 16px","background":"#1a1a2e",
                   "position":"relative","zIndex":2000},
        ),

        # Map container with legend overlay
        html.Div(
            [
                dcc.Graph(
                    id="map",
                    figure=build_base_figure(layout),
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
                        html.Hr(style={"margin":"4px 0","borderColor":"#ccc"}),
                        html.Div("Trucks (icon colour = status)",
                                 style={"fontWeight":"bold","fontSize":"12px"}),
                        html.Div([
                            html.Span("● with bikes  ", style={"color":"#6a1b9a","fontSize":"12px"}),
                            html.Span("● empty", style={"color":"#757575","fontSize":"12px"}),
                        ]),
                        html.Div([
                            html.Span("● loading/unloading  ",
                                      style={"color":"#ef6c00","fontSize":"12px"}),
                            html.Span("● idle", style={"color":"#212121","fontSize":"12px"}),
                        ]),
                        html.Div(html.Span("━ route ahead (colour per truck), faint = driven",
                                 style={"color":"#555","fontSize":"12px"})),
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
                # Every 3 hours is labelled, so a long range is wider than the page: the
                # timeline scrolls sideways and follows the play head (animation callback).
                html.Div(
                    html.Div(
                        dcc.Slider(
                            id="time-slider", min=0, max=1439, step=1, value=0,
                            marks=slider_marks(1, 0),
                            allow_direct_input=False,   # its box would show raw minutes
                            # Date and time under the cursor (assets/slider_time.js).
                            tooltip={"placement": "top", "always_visible": False,
                                     "transform": "sliderDateTime"},
                        ),
                        id="slider-inner", style=slider_width(1),
                    ),
                    id="slider-scroll",
                    style={"overflowX":"auto","overflowY":"hidden",
                           "padding":"28px 12px 4px"},    # room for the tooltip above
                ),
                html.Div("Replay of one continuous simulated run from Aug 1, 2026: stations are "
                         "filled once at 00:00 on Aug 1 and bikes carry over between days. "
                         "Running totals count from the start of the picked range.",
                         style={"fontSize":"12px","color":"#666","marginTop":"6px"}),
            ],
            style={"padding":"10px 20px","background":"#f4f4f4",
                   "borderTop":"1px solid #ddd"},
        ),
    ],
    style={"fontFamily":"sans-serif"},
)


app.layout = serve_layout


# ── Server callbacks ───────────────────────────────────────────────────────────



@app.callback(
    Output("total-trips", "children"),
    Output("range-store", "data"),
    Output("time-slider", "max"),
    Output("time-slider", "marks"),
    Output("day-select", "options"),
    Output("day-select-wrap", "style"),
    Output("slider-inner", "style"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("policy-select", "value"),
    State("day-select-wrap", "style"),
)
def on_selection_change(start, end, policy, wrap_style):
    """A new range or policy: load that policy's replay (built first if missing)."""
    if not start or not end or not policy:
        return (no_update,) * 7
    d0, d1 = _day(start), _day(end)
    if d1 < d0:
        return ("End date is before the start date",) + (no_update,) * 6
    dd.load(policy)
    n = d1 - d0 + 1
    span = _label(d0) if n == 1 else f"{_label(d0)} – {_label(d1)} ({n} days)"
    style = dict(wrap_style or {}, display="flex" if n > 1 else "none")
    return (f"{dd.requests_in(d0, d1):,} bike requests, {span}",
            {"policy": policy, "d0": d0, "n_days": n},
            n * MINUTES_PER_DAY - 1, slider_marks(n, d0),
            [{"label": _label(d0 + i, weekday=True), "value": i} for i in range(n)], style,
            slider_width(n))


@app.callback(
    Output("time-slider", "value"),
    Output("day-select", "value"),
    Input("interval", "n_intervals"),
    Input("day-select", "value"),
    Input("time-slider", "value"),
    Input("range-store", "data"),
    State("speed-select", "value"),
    prevent_initial_call=True,
)
def set_time(_, day, minute, rng, speed):
    """Keep the slider (minutes since the range start) and the day picker in step:
    playing advances the slider, picking a day jumps there at the same time of day,
    and a new range starts again at 00:00 on its first day."""
    if not rng:
        return no_update, no_update
    total, minute = rng["n_days"] * MINUTES_PER_DAY, minute or 0
    trigger = ctx.triggered_id
    if trigger == "range-store":
        return 0, 0
    if trigger == "day-select":
        if day is None or day == minute // MINUTES_PER_DAY:
            return no_update, no_update
        return day * MINUTES_PER_DAY + minute % MINUTES_PER_DAY, no_update
    if trigger == "interval":
        minute = (minute + speed) % total
        new_day = minute // MINUTES_PER_DAY
        return minute, (new_day if new_day != day else no_update)
    new_day = minute // MINUTES_PER_DAY          # slider dragged
    return no_update, (new_day if new_day != day else no_update)


@app.callback(
    Output("sim-store", "data"),
    Input("day-select", "value"),
    Input("range-store", "data"),
)
def load_day(day, rng):
    """Send the browser the replay day being shown (one day at a time, about 3 MB)."""
    if not rng:
        return no_update
    day = min(max(int(day or 0), 0), rng["n_days"] - 1)
    if day + 1 < rng["n_days"]:          # warm the next day so playback doesn't stall at midnight
        threading.Thread(target=dd.day_chunk, daemon=True,
                         args=(rng["policy"], rng["d0"] + day + 1, rng["d0"])).start()
    return dict(dd.day_chunk(rng["policy"], rng["d0"] + day, rng["d0"]), n_days=rng["n_days"])


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


# ── Clientside: animation — rebuilds the figure; uirevision keeps the view ────
app.clientside_callback(
    _ANIMATION_JS.replace("__N_OUTPUTS__", str(2 + len(METRICS))),
    Output("map", "figure"),
    Output("time-display", "children"),
    *[Output(cid, "children") for cid, _ in METRICS],
    Input("time-slider", "value"),
    Input("zone-toggle", "value"),
    Input("sim-store", "data"),
    State("station-store", "data"),
    State("map-layout", "data"),
    State("zone-store", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    print("Loading the replay for", POLICIES[DEFAULT_POLICY].label, "...")
    dd.load(DEFAULT_POLICY)
    print("Loading road network for truck routes ...")
    from road_routes import get_router
    get_router()
    print("Starting Dash at http://localhost:8050")
    app.run(debug=False, port=8050)
