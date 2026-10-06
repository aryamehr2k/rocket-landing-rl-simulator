// The right-hand panel of the Fly tab: telemetry read-out, mission checklist, verdict and live charts.

import { Chart } from './charts.js';
import { byId, el, fmt } from './dom.js';

const TELEMETRY = [
  ['t', 'Time'], ['phase', 'Phase'],
  ['height', 'Height'], ['est', 'Estimated'],
  ['vz', 'Climb rate'], ['refv', 'Plan rate'],
  ['distance', 'From pad'], ['tilt', 'Tilt'],
  ['throttle', 'Throttle'], ['thrust', 'Thrust'],
  ['gimbal', 'Gimbal p/y'], ['wind', 'Wind'],
  ['max', 'Max height'], ['rate', 'Sim speed'],
];
const CHECKS = [
  ['altitude', 'Reached the target height'],
  ['hover', 'Held the hover'],
  ['landing', 'Landed within the leg limits'],
  ['radius', 'Inside the landing radius'],
];
const CHIP_TEXT = { pass: 'PASS', fail: 'FAIL', wait: 'PENDING', run: 'LIVE' };
const RAD_TO_DEG = 180 / Math.PI;
const CALM_MPS = 0.05;

function bearingDeg(x, y) {
  return (Math.atan2(y, x) * RAD_TO_DEG + 360) % 360;
}

export class FlyPanel {
  constructor() {
    this.values = {};
    const list = byId('telemetry');
    for (const [key, label] of TELEMETRY) {
      this.values[key] = el('dd', { text: '--' });
      list.append(el('dt', { text: label }), this.values[key]);
    }
    this.items = {};
    const checklist = byId('checklist');
    for (const [key, label] of CHECKS) {
      const chip = el('span', { class: 'chip', text: CHIP_TEXT.wait });
      const detail = el('span', { class: 'detail', text: '' });
      this.items[key] = { chip, detail };
      checklist.append(el('li', {}, [chip, el('span', {}, [label, detail])]));
    }
    this.verdict = byId('verdict');
    this.buildCharts(byId('flyCharts'));
  }

  buildCharts(container) {
    const make = (title, options) => {
      const canvas = el('canvas');
      container.append(el('div', { class: 'chart' }, [el('div', { class: 'title' }, [el('span', { text: title })]), canvas]));
      return new Chart(canvas, { xMin: 0, xMinSpan: 10, xName: 't', ...options });
    };
    this.charts = {
      height: make('Height of the feet (m)', { minSpan: 4 }),
      speed: make('Vertical speed (m/s)', { minSpan: 2 }),
      throttle: make('Throttle (0 to 1)', { yMin: 0, yMax: 1 }),
      gimbal: make('Gimbal (deg)', { minSpan: 2 }),
      tilt: make('Tilt from vertical (deg)', { minSpan: 2, yMin: 0 }),
    };
  }

  // Empty charts and checklist for a new flight described by the server's setup event.
  reset(info) {
    const c = this.charts;
    for (const chart of Object.values(c)) chart.clear();
    const m = info.mission;
    c.height.bands.push({ from: m.target_altitude - m.altitude_tolerance, to: m.target_altitude + m.altitude_tolerance, color: '--band' });
    this.series = {
      height: c.height.addSeries({ name: 'flown', color: '--series-1' }),
      refHeight: c.height.addSeries({ name: 'plan', color: '--reference', dash: [5, 4] }),
      speed: c.speed.addSeries({ name: 'flown', color: '--series-1' }),
      refSpeed: c.speed.addSeries({ name: 'plan', color: '--reference', dash: [5, 4] }),
      throttle: c.throttle.addSeries({ name: 'throttle', color: '--series-1' }),
      pitch: c.gimbal.addSeries({ name: 'pitch', color: '--series-1' }),
      yaw: c.gimbal.addSeries({ name: 'yaw', color: '--series-2' }),
      tilt: c.tilt.addSeries({ name: 'tilt', color: '--series-1' }),
    };
    this.info = info;
    for (const key of Object.keys(this.items)) this.setItem(key, 'wait', '');
    this.verdict.classList.add('hidden');
    for (const value of Object.values(this.values)) value.textContent = '--';
  }

  frame(f) {
    if (!this.series) return;
    const s = this.series;
    const push = (series, y) => { series.x.push(f.t); series.y.push(y); };
    push(s.height, f.height); push(s.refHeight, f.ref_height);
    push(s.speed, f.velocity[2]); push(s.refSpeed, f.ref_speed);
    push(s.throttle, f.throttle);
    push(s.pitch, f.gimbal_deg[0]); push(s.yaw, f.gimbal_deg[1]);
    push(s.tilt, f.tilt_deg);
    for (const chart of Object.values(this.charts)) chart.requestDraw();
    this.telemetry(f);
    this.liveChecks(f);
  }

  telemetry(f) {
    const v = this.values;
    const windSpeed = Math.hypot(f.wind[0], f.wind[1]);
    v.t.textContent = fmt(f.t, 1, 's');
    v.phase.textContent = f.phase;
    v.height.textContent = fmt(f.height, 2, 'm');
    v.est.textContent = fmt(f.est_height, 2, 'm');
    v.vz.textContent = fmt(f.velocity[2], 2, 'm/s');
    v.refv.textContent = fmt(f.ref_speed, 2, 'm/s');
    v.distance.textContent = fmt(Math.hypot(f.position[0], f.position[1]), 2, 'm');
    v.tilt.textContent = fmt(f.tilt_deg, 1, 'deg');
    v.throttle.textContent = `${fmt(100 * f.throttle, 0)} %`;
    v.thrust.textContent = fmt(f.thrust, 1, 'N');
    v.gimbal.textContent = `${fmt(f.gimbal_deg[0], 1)} / ${fmt(f.gimbal_deg[1], 1)} deg`;
    v.wind.textContent = windSpeed < CALM_MPS ? 'calm' : `${fmt(windSpeed, 1)} m/s, ${fmt(bearingDeg(f.wind[0], f.wind[1]), 0)} deg`;
    v.max.textContent = fmt(f.max_height, 1, 'm');
    v.rate.textContent = Number.isFinite(f.rate) ? `${fmt(f.rate, 1)}x` : '--';
  }

  setItem(key, state, detail) {
    const { chip, detail: text } = this.items[key];
    chip.className = `chip ${state}`;
    chip.textContent = CHIP_TEXT[state];
    text.textContent = detail;
  }

  liveChecks(f) {
    const m = this.info.mission, legs = this.info.legs;
    const reached = f.max_height >= m.target_altitude - m.altitude_tolerance;
    this.setItem('altitude', reached ? 'pass' : 'wait',
                 `max ${fmt(f.max_height, 1)} m, target ${fmt(m.target_altitude, 1)} +- ${fmt(m.altitude_tolerance, 1)} m`);
    this.setItem('hover', f.hover_held >= m.hover_time ? 'pass' : 'wait',
                 `held ${fmt(f.hover_held, 1)} of ${fmt(m.hover_time, 1)} s in the band`);
    this.setItem('landing', 'wait',
                 `limits ${legs.max_vertical_speed} m/s down, ${legs.max_lateral_speed} m/s sideways, ${fmt(legs.max_tilt_deg, 0)} deg`);
    this.setItem('radius', 'wait', `now ${fmt(Math.hypot(f.position[0], f.position[1]), 1)} m from the pad, limit ${fmt(m.landing_radius, 1)} m`);
  }

  result(r) {
    const m = this.info.mission;
    const mark = (ok) => (ok ? 'pass' : 'fail');
    this.setItem('altitude', mark(r.reached_altitude), `max ${fmt(r.max_height, 1)} m, target ${fmt(m.target_altitude, 1)} +- ${fmt(m.altitude_tolerance, 1)} m`);
    this.setItem('hover', mark(r.hover_ok), `held ${fmt(r.hover_held, 1)} of ${fmt(m.hover_time, 1)} s in the band`);
    this.setItem('landing', mark(r.landed), Number.isFinite(r.touchdown_speed)
      ? `touchdown at ${fmt(r.touchdown_speed, 2)} m/s` : 'no touchdown');
    this.setItem('radius', mark(r.inside_radius), `${fmt(r.miss_distance, 1)} m from the pad, limit ${fmt(m.landing_radius, 1)} m`);
    let title = r.success ? 'Mission OK' : 'Mission failed';
    if (r.stopped) title = 'Stopped before the end';
    if (r.aborted) title += ` (aborted${r.abort_reason ? `: ${r.abort_reason}` : ''})`;
    if (r.timeout) title += ' (time limit)';
    const lines = [el('strong', { text: title }), el('span', { text: r.summary })];
    if (r.error) lines.push(el('div', { class: 'error', text: r.error }));
    this.verdict.replaceChildren(...lines);
    this.verdict.className = `verdict ${r.success ? 'pass' : 'fail'}`;
  }

  redraw() {
    for (const chart of Object.values(this.charts)) chart.requestDraw();
  }
}
