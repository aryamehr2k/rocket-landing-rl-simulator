// The Many flights tab: one setup on many seeds, optionally against the PID on the same seeds.

import { getJson, postJson } from './api.js';
import { Chart } from './charts.js';
import { byId, el, fmt, percent } from './dom.js';
import { createFlightForm } from './flightform.js';

const POLL_MS = 1000;
const PATH_WIDTH = 1.5;
const PATH_ALPHA = 0.75;

function failureReason(flight, mission) {
  if (flight.aborted) return 'aborted';
  if (flight.timeout) return 'still flying at the time limit';
  if (!flight.landed) return 'hard landing';
  if (flight.max_height < mission.target_altitude - mission.altitude_tolerance) return 'never reached the target height';
  if (flight.hover_held < mission.hover_time) return `hover held only ${fmt(flight.hover_held, 1)} s`;
  if (flight.miss > mission.landing_radius) return `landed ${fmt(flight.miss, 1)} m out`;
  return 'failed';
}

function summaryTable(results) {
  const head = ['Controller', 'Mission met', 'Landed', 'Aborted', 'Max height', 'Hover held', 'Miss', 'Touchdown'];
  const rows = Object.entries(results).map(([name, { summary: s }]) => el('tr', {}, [
    el('td', { text: name }),
    el('td', { text: `${s.success} / ${s.flights} (${percent(s.success / s.flights)})` }),
    el('td', { text: `${s.landed} / ${s.flights}` }),
    el('td', { text: String(s.aborted) }),
    el('td', { text: fmt(s.max_height, 1, 'm') }),
    el('td', { text: fmt(s.hover_held, 1, 's') }),
    el('td', { text: fmt(s.miss, 2, 'm') }),
    el('td', { text: fmt(s.touchdown_speed, 2, 'm/s') }),
  ]));
  return el('table', {}, [el('thead', {}, el('tr', {}, head.map((h) => el('th', { text: h })))), el('tbody', {}, rows)]);
}

function plotCard(title) {
  const canvas = el('canvas');
  return { card: el('div', { class: 'plot' }, [el('div', { class: 'chart' }, [el('div', { class: 'title', text: title }), canvas])]), canvas };
}

function passFailLegend(flights, extra) {
  const passed = flights.filter((f) => f.success).length;
  return [{ name: `passed (${passed})`, color: '--good' },
          { name: `failed (${flights.length - passed})`, color: '--critical' }, ...extra];
}

function controllerPlots(name, flights, mission) {
  const top = plotCard(`${name}: paths seen from above (m)`);
  const side = plotCard(`${name}: height of the feet against time (m, s)`);
  const block = el('div', {}, [el('div', { class: 'plots' }, [top.card, side.card])]);
  const failures = flights.filter((f) => !f.success);
  if (failures.length) {
    block.append(el('p', { class: 'note', text: `Failed seeds: ${failures.map((f) => `${f.seed} (${failureReason(f, mission)})`).join(', ')}` }));
  }
  const map = new Chart(top.canvas, { equalAspect: true, hover: false, minSpan: 2 * mission.landing_radius });
  const height = new Chart(side.canvas, { xMin: 0, hover: false, minSpan: 4 });
  map.circles.push({ x: 0, y: 0, r: mission.landing_radius, color: '--series-3', dash: [5, 4] });
  map.legend = passFailLegend(flights, [{ name: 'landing radius', color: '--series-3', dash: [5, 4] }]);
  height.bands.push({ from: mission.target_altitude - mission.altitude_tolerance, to: mission.target_altitude + mission.altitude_tolerance, color: '--band' });
  height.legend = passFailLegend(flights, []);
  for (const flight of flights) {
    const color = flight.success ? '--good' : '--critical';
    const style = { color, width: PATH_WIDTH, alpha: PATH_ALPHA, legend: false };
    map.addSeries({ ...style, x: flight.path.x, y: flight.path.y });
    height.addSeries({ ...style, x: flight.path.t, y: flight.path.h });
    map.points.push({ x: flight.path.x.at(-1), y: flight.path.y.at(-1), color });
  }
  return { block, charts: [map, height] };
}

export function initBatch(options) {
  const form = createFlightForm(byId('batchForm'), options, { prefix: 'batch', live: false });
  const ui = { run: byId('batchRun'), error: byId('batchError'), status: byId('batchStatus'), title: byId('batchTitle'),
               progress: byId('batchProgress'), results: byId('batchResults') };
  let charts = [];
  let shownJob = null;

  async function run() {
    ui.error.textContent = '';
    ui.run.disabled = true;
    try {
      const { id } = await postJson('/api/batch', form.values());
      poll(id);
    } catch (error) {
      ui.error.textContent = error.message;
      ui.run.disabled = false;
    }
  }

  async function poll(id) {
    try {
      const job = await getJson(`/api/batch/${id}`);
      show(job);
      if (job.state === 'running') setTimeout(() => poll(id), POLL_MS);
    } catch (error) {
      ui.error.textContent = error.message;
      ui.run.disabled = false;
    }
  }

  function show(job) {
    const s = job.setup;
    ui.title.textContent = `${job.controllers.join(' vs ')}: ${s.flights} flights from seed ${s.seed}, mission ${s.mission.name}`;
    ui.progress.classList.toggle('hidden', job.state !== 'running');
    ui.progress.firstElementChild.style.width = `${(100 * job.done) / job.total}%`;
    ui.run.disabled = job.state === 'running';
    const wind = `wind ${fmt(s.wind.speed, 1)} m/s toward ${fmt(s.wind.direction_deg, 0)} deg, gusts ${fmt(s.wind.gust_std, 1)} m/s`;
    const errors = s.random_errors ? 'hidden errors drawn per seed' : `thrust x${fmt(s.errors.thrust_scale, 2)}, mass ${fmt(s.errors.dry_mass_offset_g, 0)} g`;
    if (job.state === 'running') {
      ui.status.textContent = `Flying ${job.done} of ${job.total} on ${job.processes} processes, ${fmt(job.seconds, 0)} s so far. ${wind}; ${errors}.`;
    } else if (job.state === 'failed') {
      ui.status.textContent = 'The batch failed.';
      ui.error.textContent = job.error;
    } else {
      ui.status.textContent = `${job.total} flights in ${fmt(job.seconds, 0)} s. ${wind}; ${errors}; sensor noise x${fmt(s.errors.sensor_noise, 1)}.`;
    }
    if (job.state === 'done' && shownJob !== job.id) render(job);
  }

  function render(job) {
    shownJob = job.id;
    const blocks = [summaryTable(job.results)];
    charts = [];
    for (const [name, result] of Object.entries(job.results)) {
      const plots = controllerPlots(name, result.flights, job.setup.mission);
      blocks.push(plots.block);
      charts.push(...plots.charts);
    }
    ui.results.replaceChildren(...blocks);
    for (const chart of charts) chart.requestDraw();
  }

  ui.run.addEventListener('click', run);
  return {
    // Show the most recent batch again after a page reload.
    async restore() {
      const job = await getJson('/api/batch/latest').catch(() => null);
      if (!job || job.state === 'none') return;
      show(job);
      if (job.state === 'running') setTimeout(() => poll(job.id), POLL_MS);
    },
    selectModel(path, label) {
      form.selectController(path, label);
    },
    shown() {
      for (const chart of charts) chart.requestDraw();
    },
  };
}
