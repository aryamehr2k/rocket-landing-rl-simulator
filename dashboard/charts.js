// Plain canvas line charts: one y axis, a quiet grid, a legend and a hover read-out.
// Colours are CSS custom property names (for example '--series-1'), read at every draw so the
// charts follow the light or dark theme.

import { cssVar } from './dom.js';

const MARGIN = { left: 46, right: 12, top: 22, bottom: 20 };
const NO_LEGEND_TOP = 8;
const TICK_COUNT = 5;
const FONT = '11px system-ui, sans-serif';
const PADDING = 0.05;           // share of the data span added above and below
const LEGEND_SWATCH = 14;
const LEGEND_GAP = 12;
const TOOLTIP_PAD = 6;
const TOOLTIP_LINE = 15;
const MARKER_RADIUS = 4;

function colour(name) {
  return name.startsWith('--') ? cssVar(name) : name;
}

function niceStep(span, count) {
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const nice = unit < 1.5 ? 1 : unit < 3 ? 2 : unit < 7 ? 5 : 10;
  return nice * power;
}

function tickValues(min, max) {
  const step = niceStep(max - min, TICK_COUNT);
  const values = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
    values.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return { values, step };
}

function tickLabel(value, step) {
  if (Math.abs(value) >= 1e6) return `${+(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e4) return `${+(value / 1e3).toFixed(1)}k`;
  const digits = Math.max(0, Math.min(3, -Math.floor(Math.log10(step) + 1e-9)));
  return value.toFixed(digits);
}

// Index of the point whose x is closest to `x` in an ascending array.
function nearestIndex(xs, x) {
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}

function widen(min, max, minSpan) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  const span = Math.max(max - min, minSpan);
  const centre = (min + max) / 2;
  return [centre - span / 2, centre + span / 2];
}

export class Chart {
  // options: xMin, xMax, yMin, yMax (fixed ends or null), minSpan, xMinSpan, equalAspect, hover,
  // digits and xDigits (hover read-out), xName (hover label of x), emptyText.
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.options = { xMin: null, xMax: null, yMin: null, yMax: null, minSpan: 1, xMinSpan: 1,
                     equalAspect: false, hover: true, digits: 2, emptyText: 'no data yet', ...options };
    this.legend = null;  // [{ name, color, dash }] to replace the legend built from the series
    this.pending = false;
    this.hoverX = null;
    this.clear();
    if (this.options.hover) {
      canvas.addEventListener('mousemove', (event) => { this.hoverX = event.offsetX; this.requestDraw(); });
      canvas.addEventListener('mouseleave', () => { this.hoverX = null; this.requestDraw(); });
    }
  }

  clear() {
    this.series = [];
    this.bands = [];    // { from, to, color } horizontal bands
    this.circles = [];  // { x, y, r, color, dash }
    this.points = [];   // { x, y, color }
    this.requestDraw();
  }

  addSeries(spec) {
    const series = { name: '', color: '--series-1', width: 2, dash: [], legend: true, x: [], y: [], ...spec };
    this.series.push(series);
    return series;
  }

  requestDraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  legendItems() {
    if (this.legend) return this.legend;
    const items = this.series.filter((s) => s.legend);
    return items.length > 1 ? items : [];
  }

  bounds(plot) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of this.series) {
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y === null || !Number.isFinite(y)) continue;
        xMin = Math.min(xMin, s.x[i]); xMax = Math.max(xMax, s.x[i]);
        yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
      }
    }
    for (const band of this.bands) { yMin = Math.min(yMin, band.from); yMax = Math.max(yMax, band.to); }
    for (const c of this.circles) {
      xMin = Math.min(xMin, c.x - c.r); xMax = Math.max(xMax, c.x + c.r);
      yMin = Math.min(yMin, c.y - c.r); yMax = Math.max(yMax, c.y + c.r);
    }
    const o = this.options;
    const x = widen(o.xMin ?? xMin, o.xMax ?? xMax, o.xMinSpan);
    let y = widen(yMin, yMax, o.minSpan);
    if (!x || !y) return null;
    const pad = (y[1] - y[0]) * PADDING;
    y = [o.yMin ?? y[0] - pad, o.yMax ?? y[1] + pad];
    if (o.equalAspect) return this.squareUp(x, y, plot);
    return { xMin: x[0], xMax: x[1], yMin: y[0], yMax: y[1] };
  }

  // Same metres per pixel on both axes, keeping every point inside.
  squareUp(x, y, plot) {
    const width = plot.right - plot.left, height = plot.bottom - plot.top;
    const scale = Math.max((x[1] - x[0]) / width, (y[1] - y[0]) / height);
    const cx = (x[0] + x[1]) / 2, cy = (y[0] + y[1]) / 2;
    return { xMin: cx - scale * width / 2, xMax: cx + scale * width / 2,
             yMin: cy - scale * height / 2, yMax: cy + scale * height / 2 };
  }

  draw() {
    const canvas = this.canvas;
    const width = Math.round(canvas.clientWidth), height = Math.round(canvas.clientHeight);
    if (width === 0 || height === 0) return;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = FONT;
    const legend = this.legendItems();
    const plot = { left: MARGIN.left, right: width - MARGIN.right, bottom: height - MARGIN.bottom,
                   top: legend.length ? MARGIN.top : NO_LEGEND_TOP };
    const b = this.bounds(plot);
    if (!b) {
      ctx.fillStyle = cssVar('--muted');
      ctx.fillText(this.options.emptyText, plot.left, (plot.top + plot.bottom) / 2);
      return;
    }
    const sx = (x) => plot.left + (x - b.xMin) / (b.xMax - b.xMin) * (plot.right - plot.left);
    const sy = (y) => plot.bottom - (y - b.yMin) / (b.yMax - b.yMin) * (plot.bottom - plot.top);
    this.drawAxes(ctx, plot, b, sx, sy);
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
    ctx.clip();
    for (const band of this.bands) {
      ctx.fillStyle = colour(band.color);
      ctx.fillRect(plot.left, sy(band.to), plot.right - plot.left, sy(band.from) - sy(band.to));
    }
    for (const c of this.circles) this.drawCircle(ctx, c, sx, sy);
    for (const s of this.series) this.drawSeries(ctx, s, sx, sy);
    for (const p of this.points) this.drawPoint(ctx, p, sx, sy);
    ctx.restore();
    this.drawLegend(ctx, legend, plot);
    if (this.hoverX !== null && this.hoverX >= plot.left && this.hoverX <= plot.right) {
      this.drawHover(ctx, plot, b, sy);
    }
  }

  drawAxes(ctx, plot, b, sx, sy) {
    const xt = tickValues(b.xMin, b.xMax), yt = tickValues(b.yMin, b.yMax);
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    ctx.fillStyle = cssVar('--muted');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt.values) {
      const y = Math.round(sy(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y); ctx.stroke();
      ctx.fillText(tickLabel(v, yt.step), plot.left - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xt.values) {
      const x = Math.round(sx(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke();
      ctx.fillText(tickLabel(v, xt.step), x, plot.bottom + 5);
    }
    ctx.strokeStyle = cssVar('--axis');
    ctx.beginPath(); ctx.moveTo(plot.left, plot.bottom + 0.5); ctx.lineTo(plot.right, plot.bottom + 0.5); ctx.stroke();
  }

  drawSeries(ctx, s, sx, sy) {
    ctx.strokeStyle = colour(s.color);
    ctx.lineWidth = s.width;
    ctx.lineJoin = 'round';
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.setLineDash(s.dash);
    ctx.beginPath();
    let drawing = false;
    for (let i = 0; i < s.x.length; i++) {
      const y = s.y[i];
      if (y === null || !Number.isFinite(y)) { drawing = false; continue; }
      if (drawing) ctx.lineTo(sx(s.x[i]), sy(y)); else ctx.moveTo(sx(s.x[i]), sy(y));
      drawing = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  drawCircle(ctx, c, sx, sy) {
    ctx.strokeStyle = colour(c.color);
    ctx.lineWidth = 1.5;
    ctx.setLineDash(c.dash || []);
    ctx.beginPath();
    ctx.arc(sx(c.x), sy(c.y), Math.abs(sx(c.x + c.r) - sx(c.x)), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawPoint(ctx, p, sx, sy) {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), MARKER_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = colour(p.color);
    ctx.strokeStyle = cssVar('--surface');
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }

  drawLegend(ctx, items, plot) {
    let x = plot.left;
    const y = 9;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const item of items) {
      ctx.strokeStyle = colour(item.color);
      ctx.lineWidth = 2;
      ctx.setLineDash(item.dash || []);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + LEGEND_SWATCH, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = cssVar('--text-2');
      ctx.fillText(item.name, x + LEGEND_SWATCH + 4, y);
      x += LEGEND_SWATCH + 4 + ctx.measureText(item.name).width + LEGEND_GAP;
    }
  }

  drawHover(ctx, plot, b, sy) {
    const x = b.xMin + (this.hoverX - plot.left) / (plot.right - plot.left) * (b.xMax - b.xMin);
    const named = this.series.filter((s) => s.legend && s.x.length);
    if (!named.length) return;
    const lines = [`${this.options.xName ?? 'x'} ${x.toFixed(this.options.xDigits ?? 1)}`];
    const marks = [];
    for (const s of named) {
      const i = nearestIndex(s.x, x);
      const y = s.y[i];
      lines.push(`${s.name || 'value'} ${Number.isFinite(y) ? y.toFixed(this.options.digits) : '--'}`);
      if (Number.isFinite(y)) marks.push({ y: sy(y), color: colour(s.color) });
    }
    ctx.strokeStyle = cssVar('--axis');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(this.hoverX + 0.5, plot.top); ctx.lineTo(this.hoverX + 0.5, plot.bottom); ctx.stroke();
    for (const m of marks) {
      ctx.beginPath(); ctx.arc(this.hoverX, m.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = m.color; ctx.fill();
    }
    const boxWidth = Math.max(...lines.map((line) => ctx.measureText(line).width)) + 2 * TOOLTIP_PAD;
    const boxHeight = lines.length * TOOLTIP_LINE + TOOLTIP_PAD;
    const left = this.hoverX + boxWidth + 12 > plot.right ? this.hoverX - boxWidth - 8 : this.hoverX + 8;
    ctx.fillStyle = cssVar('--surface');
    ctx.strokeStyle = cssVar('--border');
    ctx.fillRect(left, plot.top, boxWidth, boxHeight);
    ctx.strokeRect(left + 0.5, plot.top + 0.5, boxWidth, boxHeight);
    ctx.fillStyle = cssVar('--text');
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    lines.forEach((line, i) => ctx.fillText(line, left + TOOLTIP_PAD, plot.top + TOOLTIP_PAD / 2 + i * TOOLTIP_LINE + 2));
  }
}
