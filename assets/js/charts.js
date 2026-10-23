// Chart motion and hover readouts for the blog. The charts are plain HTML and read fine without this script:
// it only animates them in when they scroll into view, counts up the headline numbers, and shows a tooltip.
(function () {
  window.chartsReady = true;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var charts = document.querySelectorAll('.chart');

  function countUp(fig) {
    fig.querySelectorAll('.kpi-value[data-count]').forEach(function (el) {
      var end = parseFloat(el.dataset.count), dec = +el.dataset.dec, suffix = el.dataset.suffix || '';
      var fmt = function (v) {
        return v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
      };
      if (reduce) { el.textContent = fmt(end); return; }
      var t0 = null, dur = 1200;
      function step(t) {
        if (t0 === null) t0 = t;
        var p = Math.min(1, (t - t0) / dur);
        el.textContent = fmt(end * (1 - Math.pow(1 - p, 3)));
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  function show(fig) { fig.classList.add('in-view'); countUp(fig); }

  if (reduce || !('IntersectionObserver' in window)) {
    charts.forEach(show);
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { show(en.target); io.unobserve(en.target); }
      });
    }, { threshold: 0.35 });
    charts.forEach(function (c) { io.observe(c); });
  }

  // One tooltip for every chart row: the value first, the label under it. Text only, set with textContent.
  var tip = document.createElement('div'), tipValue = document.createElement('b'), tipLabel = document.createElement('span');
  tip.className = 'chart-tip';
  tip.appendChild(tipValue);
  tip.appendChild(tipLabel);
  document.body.appendChild(tip);

  function fill(row) {
    tipValue.textContent = row.dataset.tipValue;
    tipLabel.textContent = row.dataset.tipLabel;
    tip.classList.add('on');
  }
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.max(8, Math.min(x + 14, window.innerWidth - w - 8));
    var top = y - h - 12 < 8 ? y + 18 : y - h - 12;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hide() { tip.classList.remove('on'); }

  document.querySelectorAll('.chart .row[data-tip-value]').forEach(function (row) {
    row.addEventListener('pointermove', function (e) { fill(row); place(e.clientX, e.clientY); });
    row.addEventListener('pointerleave', hide);
    row.addEventListener('focus', function () {
      fill(row);
      var r = row.querySelector('.track').getBoundingClientRect();
      place(r.left + r.width / 2, r.top);
    });
    row.addEventListener('blur', hide);
  });
  window.addEventListener('scroll', hide, { passive: true });
})();
