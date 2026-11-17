// Bike Share Toronto: static replay viewer (GitHub Pages copy of app.py, no server).
// Reads the files written by deploy/build_static.py and draws the same map, minute by
// minute, as the Dash app's animation callback.
(function () {
'use strict';

var M = 1440;                       // minutes per day
var PX_PER_DAY = 300;               // timeline width per day, so 3-hour labels never overlap
var REPO = 'https://github.com/zhangchi9/toronto_bike_share_rl';
var $ = function (id) { return document.getElementById(id); };

// ── Data files ─────────────────────────────────────────────────────────────────
async function fetchBytes(url) {
    var r = await fetch(url);
    if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
    var buf = new Uint8Array(await r.arrayBuffer());
    if (buf[0] === 0x1f && buf[1] === 0x8b) {        // gzip (unless the server unzipped it)
        var ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
        return new Uint8Array(await new Response(ds).arrayBuffer());
    }
    return buf;
}

async function fetchJson(url) {
    return JSON.parse(new TextDecoder().decode(await fetchBytes(url)));
}

var DTYPES = {'|u1': Uint8Array, '|i1': Int8Array, '<u2': Uint16Array, '<i2': Int16Array,
              '<u4': Uint32Array, '<i4': Int32Array, '<f8': Float64Array};

async function fetchBin(url) {      // see deploy/build_static.py for the layout
    var b = await fetchBytes(url);
    var buf = b.byteOffset === 0 && b.byteLength === b.buffer.byteLength ? b.buffer
              : b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    var hlen = new DataView(buf).getUint32(0, true);
    var head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    var base = 4 + hlen, out = {info: head.info};
    Object.keys(head.arrays).forEach(function (name) {
        var a = head.arrays[name], T = DTYPES[a[0]];
        if (!T) throw new Error(url + ': unknown dtype ' + a[0]);
        out[name] = new T(buf, base + a[1], a[2]);
    });
    return out;
}

// ── State ──────────────────────────────────────────────────────────────────────
var S = {
    meta: null, stations: null, zones: [], routes: null, routesP: null,
    policy: null, d0: 0, nDays: 7, cur: 0, timer: null,
    days: new Map(),        // "policy|d" -> promise of a prepared day
    trips: new Map(),       // d -> promise of that day's rides
    ready: new Map(),       // "policy|d" -> prepared day, once loaded
    facing: [], layout: null
};

function remember(map, key, make, keep) {
    if (!map.has(key)) {
        map.set(key, make());
        while (map.size > keep) map.delete(map.keys().next().value);   // oldest first
    }
    return map.get(key);
}

function tripsDay(d) {
    return remember(S.trips, d, function () {
        return fetchBin('data/trips/' + d + '.bin.gz');
    }, 6);
}

function routes() {
    if (!S.routesP) S.routesP = fetchJson('data/routes.json.gz').then(function (r) {
        S.routes = r; return r;
    });
    return S.routesP;
}

// One policy's day d, with what the drawing needs precomputed.
function getDay(policy, d) {
    var key = policy + '|' + d;
    var p = remember(S.days, key, async function () {
        var res = await Promise.all([fetchBin('data/' + policy + '/' + d + '.bin.gz'),
                                     tripsDay(d),
                                     S.meta.policies[policy].n_trucks ? routes() : null]);
        var pol = res[0], day = {key: key, policy: policy, d: d, pol: pol, trips: res[1]};
        var off = new Int32Array(M + 1);                 // changes of minute m: off[m]..off[m+1]
        for (var m = 0; m < M; m++) off[m + 1] = off[m] + pol.chg_count[m];
        day.off = off;
        day.state = new Uint8Array(pol.bikes0);
        day.stateMin = 0;
        day.nTrucks = pol.info.n_trucks;
        if (day.nTrucks) day.legsBy = truckLegs(pol);
        return day;
    }, 8);
    p.then(function (day) {
        S.ready.set(key, day);
        while (S.ready.size > 8) S.ready.delete(S.ready.keys().next().value);
    }, function () { S.days.delete(key); });
    return p;
}

// Per truck: its trips in time order, with cumulative distance along the road route.
function truckLegs(pol) {
    var legsBy = [], L = pol.legs, st = S.stations;
    for (var q = 0; q < pol.info.n_trucks; q++) legsBy.push([]);
    for (var i = 0; i < L.length; i += 6) {
        var frm = L[i + 1], to = L[i + 2];
        var r = S.routes[frm + ',' + to] || [[], []];
        var lat = r[0], lon = r[1];
        if (lat.length < 2) { lat = [st.lat[frm], st.lat[to]]; lon = [st.lon[frm], st.lon[to]]; }
        var cum = [0];
        for (var p = 1; p < lat.length; p++) {
            var dy = lat[p] - lat[p - 1], dx = (lon[p] - lon[p - 1]) * 0.72;
            cum.push(cum[p - 1] + Math.sqrt(dx * dx + dy * dy));
        }
        legsBy[L[i]].push({k: L[i], to: to, depart: L[i + 3], arrive: L[i + 4], load: L[i + 5],
                           lat: lat, lon: lon, cum: cum});
    }
    return legsBy;
}

function bikesAt(day, t) {
    var pol = day.pol;
    if (t < day.stateMin) { day.state.set(pol.bikes0); day.stateMin = 0; }
    for (var m = day.stateMin + 1; m <= t; m++) {
        for (var c = day.off[m]; c < day.off[m + 1]; c++) day.state[pol.chg_station[c]] = pol.chg_value[c];
    }
    day.stateMin = t;
    return day.state;
}

// Running totals since 00:00 on the first day of the picked range.
function cumulative(day) {
    if (day.cum && day.cum.d0 === S.d0) return day.cum;
    var sums = S.meta.policies[day.policy].day_sums, cum = {d0: S.d0};
    ['reward', 'fulfilled', 'lost', 'blocked'].forEach(function (k) {
        var acc = 0, arr = new Float64Array(M), src = day.pol[k];
        for (var j = S.d0; j < day.d; j++) acc += sums[k][j];
        for (var m = 0; m < M; m++) { acc += src[m]; arr[m] = acc; }
        cum[k] = arr;
    });
    day.cum = cum;
    return cum;
}

// ── Dates ──────────────────────────────────────────────────────────────────────
function dayDate(d) {
    var s = S.meta.start.split('-');
    return new Date(Date.UTC(+s[0], +s[1] - 1, +s[2] + d));
}
function label(d, weekday) {
    var o = {timeZone: 'UTC', month: 'short', day: 'numeric'};
    if (weekday) o.weekday = 'short';
    return dayDate(d).toLocaleDateString('en-US', o).replace(',', '');
}
function dayOf(iso) {
    var s = iso.split('-'), t = S.meta.start.split('-');
    return Math.round((Date.UTC(+s[0], +s[1] - 1, +s[2]) - Date.UTC(+t[0], +t[1] - 1, +t[2])) / 864e5);
}
function hourLabel(h) { return h === 12 ? '12pm' : (h % 12) + (h < 12 ? 'am' : 'pm'); }
function clock(m) {
    return String(Math.floor(m / 60) % 24).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
}

// ── Map ────────────────────────────────────────────────────────────────────────
// Basemap: OSM raster tiles drawn faded over white so stations and rides stand out.
function mapLayout() {
    var base = new URL('.', location.href).href;
    return {
        map: {
            style: {
                version: 8, id: 'faded-osm-trucks',
                sprite: base + 'assets/truck_sprite/sprite',
                glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
                sources: {osm: {type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                                tileSize: 256, maxzoom: 19,
                                attribution: '© OpenStreetMap contributors'}},
                layers: [
                    {id: 'background', type: 'background', paint: {'background-color': '#ffffff'}},
                    {id: 'osm', type: 'raster', source: 'osm',
                     paint: {'raster-opacity': 0.45, 'raster-saturation': -0.3}}
                ]
            },
            center: {lat: 43.655, lon: -79.38}, zoom: 11
        },
        margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false, uirevision: 'keep-view'
    };
}

var METRICS = [
    ['m-reward', 'Reward this minute'],
    ['m-cum-reward', 'Cumulative reward'],
    ['m-fulfilled', 'Requests fulfilled this minute'],
    ['m-cum-fulfilled', 'Total requests fulfilled'],
    ['m-cum-lost', 'Total requests lost'],
    ['m-active', 'Active trips'],
    ['m-blocked', 'Returns blocked this minute (station full)'],
    ['m-cum-blocked', 'Total returns blocked'],
    ['m-pending', 'Bikes pending (held 15 min)'],
    ['m-trucks', 'Trucks'],
    ['m-truck-moved', 'Bikes moved by trucks']
];
var ZC = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39',
          '#7b4173', '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363',
          '#111111'];   // 21st: the island zone
var TRUCK_COL = ['#7b1fa2', '#00838f', '#ad1457', '#2e7d32'];   // route line per truck
var ICON = {drive_full: 'full', drive_empty: 'empty', handling: 'handling', idle: 'idle'};

function money(x) {
    return (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString('en-US',
        {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function num(x) { return Math.round(x).toLocaleString('en-US'); }

function zoneTraces() {
    var traces = [], labels = [];
    if (!S.zones.length || !$('zone-toggle').checked) return [traces, labels];
    S.zones.forEach(function (z, i) {
        var c = ZC[i % ZC.length];
        var r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
        traces.push({type: 'scattermap', mode: 'lines', fill: 'toself',
            fillcolor: 'rgba(' + r + ',' + g + ',' + b + ',0.08)', lat: z.lat, lon: z.lon,
            line: {color: c, width: 2}, hoverinfo: 'text',
            text: 'Zone ' + z.zone + ': ' + z.stations + ' stations, ' + z.docks.toLocaleString('en-US') + ' docks'});
    });
    labels.push({type: 'scattermap', mode: 'text', hoverinfo: 'skip',
        lat: S.zones.map(function (z) { return z.clat; }),
        lon: S.zones.map(function (z) { return z.clon; }),
        text: S.zones.map(function (z) { return 'Z' + z.zone; }),
        textfont: {size: 15, color: '#111111', family: 'Noto Sans Bold'}});
    return [traces, labels];
}

function setText(id, s) { var el = $(id); if (el.textContent !== s) el.textContent = s; }

var drawing = false, redraw = false;
function render() {
    if (!S.meta) return;
    var i = Math.floor(S.cur / M), d = S.d0 + i, t = S.cur - i * M;
    var day = S.ready.get(S.policy + '|' + d);
    if (!day) {                                       // not loaded yet: draw when it is
        $('loading').style.display = 'block';
        getDay(S.policy, d).then(render, function (e) {
            $('loading').textContent = 'Could not load ' + label(d) + ': ' + e.message;
        });
        return;
    }
    $('loading').style.display = 'none';
    if (i + 1 < S.nDays) getDay(S.policy, d + 1);     // so playback doesn't stall at midnight

    var pol = day.pol, tr = day.trips, st = S.stations, ns = st.lat.length;
    var bikes = bikesAt(day, t), cum = cumulative(day);

    var color = new Array(ns), size = new Array(ns), text = new Array(ns);
    for (var k = 0; k < ns; k++) {
        var nb = bikes[k], cap = st.cap[k] || 1;
        color[k] = nb / cap;
        size[k] = 5 + 10 * Math.sqrt(cap / 90);
        text[k] = st.name[k] + '<br>bikes: ' + nb + ' / ' + st.cap[k] + ' docks';
    }

    // Rides in progress (only requests the simulation actually served).
    var mLat = [], mLon = [], cLat = [], cLon = [];
    for (var j = 0; j < tr.sm.length; j++) {
        if (!pol.trip_ok[j] || tr.sm[j] > t || tr.em[j] <= t) continue;
        var s = tr.s[j], e = tr.e[j];
        if (tr.casual[j]) { cLat.push(st.lat[s], st.lat[e], null); cLon.push(st.lon[s], st.lon[e], null); }
        else              { mLat.push(st.lat[s], st.lat[e], null); mLon.push(st.lon[s], st.lon[e], null); }
    }

    // Trucks: position along the road route of the trip in progress.
    var truckTraces = [], truckCard = 'No trucks', movedCard = '–';
    if (day.nTrucks) {
        var icons = {lat: [], lon: [], symbol: [], text: []}, parts = [];
        for (var q = 0; q < day.nTrucks; q++) {
            var stt = pol.truck_status[q * M + t], load = pol.truck_load[q * M + t];
            var at = pol.truck_station[q * M + t];
            var plat = st.lat[at], plon = st.lon[at], desc, state, leg = null;
            if (stt === 1) {
                var Ls = day.legsBy[q];
                for (var x = Ls.length - 1; x >= 0; x--) {
                    if (Ls[x].depart <= t && t < Ls[x].arrive) { leg = Ls[x]; break; }
                }
            }
            if (leg) {
                var f = Math.min(1, (t + 1 - leg.depart) / Math.max(1, leg.arrive - leg.depart));
                var target = f * leg.cum[leg.cum.length - 1], p = 1;
                while (p < leg.cum.length - 1 && leg.cum[p] < target) p++;
                var seg = leg.cum[p] - leg.cum[p - 1], w = seg > 0 ? (target - leg.cum[p - 1]) / seg : 1;
                plat = leg.lat[p - 1] + w * (leg.lat[p] - leg.lat[p - 1]);
                plon = leg.lon[p - 1] + w * (leg.lon[p] - leg.lon[p - 1]);
                var dx = leg.lon[p] - leg.lon[p - 1];         // face the direction of travel
                if (Math.abs(dx) > 1e-6) S.facing[q] = dx > 0 ? 'r' : 'l';
                var col = TRUCK_COL[q % TRUCK_COL.length];
                truckTraces.push({type: 'scattermap', mode: 'lines', hoverinfo: 'skip',
                    lat: leg.lat.slice(0, p).concat([plat]), lon: leg.lon.slice(0, p).concat([plon]),
                    line: {color: col, width: 3}, opacity: 0.35});
                truckTraces.push({type: 'scattermap', mode: 'lines', hoverinfo: 'skip',
                    lat: [plat].concat(leg.lat.slice(p)), lon: [plon].concat(leg.lon.slice(p)),
                    line: {color: col, width: 5}, opacity: 0.9});
                state = leg.load > 0 ? 'drive_full' : 'drive_empty';
                desc = 'driving ' + (leg.load > 0 ? leg.load + ' bikes' : 'empty') + ' to '
                       + st.name[leg.to] + ', arrives ' + clock(leg.arrive);
                parts.push('T' + (q + 1) + ': → ' + st.name[leg.to]
                           + (leg.load > 0 ? ' (' + leg.load + ' bikes)' : ' (empty)'));
            } else if (stt === 2) {
                state = 'handling';
                desc = 'loading / unloading at ' + st.name[at] + ' (' + load + ' bikes on board)';
                parts.push('T' + (q + 1) + ': loading/unloading at ' + st.name[at]);
            } else {
                state = 'idle';
                desc = 'idle at ' + st.name[at];
                parts.push('T' + (q + 1) + ': idle');
            }
            icons.lat.push(plat); icons.lon.push(plon);
            icons.symbol.push('truck-' + ICON[state] + '-' + (S.facing[q] || 'r'));
            icons.text.push('<b>Truck ' + (q + 1) + '</b><br>' + desc);
        }
        truckTraces.push({type: 'scattermap', mode: 'markers', hoverinfo: 'text',
            lat: icons.lat, lon: icons.lon, text: icons.text,
            marker: {symbol: icons.symbol, size: 14, allowoverlap: true}});
        truckCard = parts.join(' · ');
        var base = S.d0 > 0 ? S.meta.policies[S.policy].moved_end[S.d0 - 1] : 0;
        movedCard = num(pol.truck_moved[t] - base);
    }

    var requests = cum.fulfilled[t] + cum.lost[t];
    var pct = requests ? ' (' + (100 * cum.fulfilled[t] / requests).toFixed(1) + '%)' : '';
    var values = [
        money(pol.reward[t]),
        money(cum.reward[t]),
        pol.fulfilled[t] + ' of ' + (pol.fulfilled[t] + pol.lost[t]),
        num(cum.fulfilled[t]) + pct,
        num(cum.lost[t]),
        num(pol.active[t]),
        num(pol.blocked[t]),
        num(cum.blocked[t]),
        num(pol.pending[t]),
        truckCard,
        movedCard
    ];
    setText('time-display', label(d) + '  ' + clock(t));
    METRICS.forEach(function (m, n) { setText(m[0], values[n]); });

    // Carry over the live camera so Plotly.react never moves the viewport; mid-gesture
    // (wheel zoom easing, drag inertia) skip the redraw so the camera isn't cut short.
    var gd = $('map'), layout = Object.assign({}, S.layout);
    var sub = gd._fullLayout && gd._fullLayout.map && gd._fullLayout.map._subplot;
    if (sub && sub.map) {
        if (sub.map.isMoving()) return;
        var c = sub.map.getCenter();
        layout.map = Object.assign({}, S.layout.map, {center: {lat: c.lat, lon: c.lng},
            zoom: sub.map.getZoom(), bearing: sub.map.getBearing(), pitch: sub.map.getPitch()});
    }
    if (drawing) { redraw = true; return; }           // draw again when this frame is done
    var z = zoneTraces();
    var data = z[0].concat([
        {type: 'scattermap', mode: 'lines', lat: mLat, lon: mLon,
         line: {color: 'rgba(51,153,255,0.5)', width: 1}, hoverinfo: 'skip'},
        {type: 'scattermap', mode: 'lines', lat: cLat, lon: cLon,
         line: {color: 'rgba(255,136,51,0.5)', width: 1}, hoverinfo: 'skip'},
        {type: 'scattermap', mode: 'markers', lat: st.lat, lon: st.lon,
         marker: {color: color, size: size, cmin: 0, cmax: 1, opacity: 0.9,
                  colorscale: [[0, '#e8000b'], [0.15, '#ff9900'], [0.5, '#2ca02c'], [1, '#1f4fd1']]},
         text: text, hoverinfo: 'text'}
    ]).concat(z[1]).concat(truckTraces);
    drawing = true;
    Promise.resolve(Plotly.react(gd, data, layout, {scrollZoom: true, displayModeBar: false,
                                                    responsive: true}))
        .finally(function () {
            drawing = false;
            if (redraw) { redraw = false; render(); }
        });
}

// ── Timeline ───────────────────────────────────────────────────────────────────
function buildTimeline() {
    var n = S.nDays, max = n * M - 1, slider = $('time-slider'), marks = $('slider-marks');
    slider.max = max;
    $('slider-inner').style.width = 'max(100%, ' + (n * PX_PER_DAY) + 'px)';
    var html = [];
    for (var i = 0; i < n; i++) {
        html.push('<span class="date" style="left:' + (100 * i * M / max) + '%">' + label(S.d0 + i) + '</span>');
        for (var h = 3; h < 24; h += 3) {
            html.push('<span style="left:' + (100 * (i * M + h * 60) / max) + '%">' + hourLabel(h) + '</span>');
        }
    }
    marks.innerHTML = html.join('');
    var sel = $('day-select');
    sel.innerHTML = '';
    for (var k = 0; k < n; k++) sel.add(new Option(label(S.d0 + k, true), k));
    $('day-select-wrap').hidden = n <= 1;
}

function showTime() {                       // slider, day picker and play head in step
    var slider = $('time-slider');
    if (+slider.value !== S.cur) slider.value = S.cur;
    var day = Math.floor(S.cur / M);
    if (+$('day-select').value !== day) $('day-select').value = day;
    var sc = $('slider-scroll');                      // long ranges scroll sideways
    if (sc.scrollWidth > sc.clientWidth + 1) {
        var pad = 12 + 28, track = sc.scrollWidth - 2 * pad;
        var x = pad + track * S.cur / Math.max(1, S.nDays * M - 1);
        if (x < sc.scrollLeft + 60 || x > sc.scrollLeft + sc.clientWidth - 60) {
            sc.scrollLeft = Math.max(0, x - sc.clientWidth * 0.25);
        }
    }
    render();
}

var tipTimer = null;
function tipAt(value, clientX) {
    var tip = $('slider-tip'), rect = $('slider-inner').getBoundingClientRect();
    clearTimeout(tipTimer);
    tipTimer = setTimeout(function () { tip.hidden = true; }, 1500);   // e.g. after a drag
    var t = Math.max(0, Math.round(value));
    tip.textContent = label(S.d0 + Math.floor(t / M), true) + ' ' + clock(t % M);
    tip.style.left = (clientX - rect.left) + 'px';
    tip.hidden = false;
}

function sliderX(value) {
    var r = $('time-slider').getBoundingClientRect(), th = 16;
    return r.left + th / 2 + (r.width - th) * value / Math.max(1, S.nDays * M - 1);
}

// ── Controls ───────────────────────────────────────────────────────────────────
function onSelection(keepTime) {
    var a = $('date-start').value, b = $('date-end').value;
    if (!a || !b) return;
    var d0 = dayOf(a), d1 = dayOf(b);
    var last = S.meta.n_days - 1;
    if (d0 < 0 || d1 > last || d0 > last || d1 < 0) {
        setText('total-trips', 'Pick dates between ' + label(0) + ' and ' + label(last));
        return;
    }
    if (d1 < d0) { setText('total-trips', 'End date is before the start date'); return; }
    S.policy = $('policy-select').value;
    var sameRange = d0 === S.d0 && d1 - d0 + 1 === S.nDays;
    S.d0 = d0;
    S.nDays = d1 - d0 + 1;
    var req = 0;
    for (var d = d0; d <= d1; d++) req += S.meta.requests[d];
    var span = S.nDays === 1 ? label(d0) : label(d0) + ' – ' + label(d1) + ' (' + S.nDays + ' days)';
    setText('total-trips', num(req) + ' bike requests, ' + span);
    if (!(keepTime && sameRange)) {
        S.cur = 0;
        buildTimeline();
    }
    S.facing = [];
    showTime();
}

function setPlaying(on) {
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    $('play-btn').textContent = on ? '■ Stop' : '▶ Play';
    if (!on) return;
    S.timer = setInterval(function () {
        var d = S.d0 + Math.floor(S.cur / M);
        if (!S.ready.has(S.policy + '|' + d)) return;          // wait for the day to load
        S.cur = (S.cur + (+$('speed-select').value)) % (S.nDays * M);
        showTime();
    }, 500);
}

function wire() {
    $('date-start').addEventListener('change', function () { onSelection(false); });
    $('date-end').addEventListener('change', function () { onSelection(false); });
    $('policy-select').addEventListener('change', function () { onSelection(true); });
    $('zone-toggle').addEventListener('change', render);
    $('day-select').addEventListener('change', function () {
        S.cur = (+this.value) * M + S.cur % M;
        showTime();
    });
    $('play-btn').addEventListener('click', function () { setPlaying(!S.timer); });
    var slider = $('time-slider');
    slider.addEventListener('input', function () {
        S.cur = +slider.value;
        tipAt(S.cur, sliderX(S.cur));
        showTime();
    });
    slider.addEventListener('pointermove', function (ev) {
        var r = slider.getBoundingClientRect(), th = 16;
        var frac = Math.min(1, Math.max(0, (ev.clientX - r.left - th / 2) / (r.width - th)));
        tipAt(frac * (S.nDays * M - 1), ev.clientX);
    });
    slider.addEventListener('pointerleave', function () { $('slider-tip').hidden = true; });
    slider.addEventListener('change', function () { $('slider-tip').hidden = true; });
}

async function init() {
    var cards = METRICS.map(function (m) {
        return '<div class="card"><div class="lbl">' + m[1] + '</div><div class="val" id="'
               + m[0] + '">–</div></div>';
    });
    $('metrics').innerHTML = cards.join('');
    if ($('legend') && window.matchMedia('(max-width: 600px)').matches) $('legend').open = false;
    $('loading').style.display = 'block';
    try {
        var r = await Promise.all([fetchJson('data/meta.json.gz'), fetchJson('data/stations.json.gz'),
                                   fetchJson('data/zones.json.gz').catch(function () { return []; })]);
        S.meta = r[0]; S.stations = r[1]; S.zones = r[2];
    } catch (e) {
        $('loading').textContent = 'Could not load the replay data: ' + e.message;
        return;
    }
    var sel = $('policy-select');
    S.meta.order.forEach(function (k) { sel.add(new Option(S.meta.policies[k].label, k)); });
    ['date-start', 'date-end'].forEach(function (id) {
        $(id).min = S.meta.start; $(id).max = S.meta.end;
    });
    $('footer').innerHTML = 'Static copy of the project dashboard' +
        (S.meta.commit ? ', built from commit <code>' + S.meta.commit + '</code>' : '') +
        '. <a href="' + REPO + '">Source code</a>. Map tiles © OpenStreetMap contributors.';
    S.layout = mapLayout();
    S.d0 = -1;                                         // force a fresh timeline
    wire();
    onSelection(false);
}

init();
})();
