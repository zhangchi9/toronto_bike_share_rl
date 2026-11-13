// Time-slider tooltip: the slider value counts minutes from 00:00 on the first day of
// the picked range. The animation callback stores that day as window._bsRangeStart.
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.sliderDateTime = function (value) {
    var start = window._bsRangeStart;
    var t = Math.max(0, Math.round(value || 0));
    var hh = String(Math.floor(t / 60) % 24).padStart(2, '0');
    var mm = String(t % 60).padStart(2, '0');
    if (!start) return hh + ':' + mm;
    var d = new Date(start + 'T00:00:00');
    d.setDate(d.getDate() + Math.floor(t / 1440));
    return d.toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric'})
        + ' ' + hh + ':' + mm;
};
