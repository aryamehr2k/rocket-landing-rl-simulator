// The Training tab: run folders with their learning curves and console logs, and starting a new training.

import { getJson, postJson } from './api.js';
import { Chart } from './charts.js';
import { ago, byId, compact, el, fmt, percent } from './dom.js';

const REFRESH_MS = 10000;
const DEFAULT_TRAINING = 'configs/training/hop.yaml';
const CURVES = [
  { key: 'success', title: 'Mission success rate (last 100 flights)', yMin: 0, yMax: 1 },
  { key: 'landed', title: 'Landed rate (last 100 flights)', yMin: 0, yMax: 1 },
  { key: 'reward', title: 'Mean episode reward', minSpan: 1 },
  { key: 'miss', title: 'Mean miss distance (m)', yMin: 0, minSpan: 1 },
  { key: 'wind', title: 'Curriculum: strongest wind (m/s)', yMin: 0, minSpan: 2 },
  { key: 'touchdown_speed', title: 'Mean touchdown speed (m/s)', yMin: 0, minSpan: 1 },
];
const YAML_WIND = -1;  // logged when a curriculum stage keeps the training file's own wind

function runRow(run, selected, onSelect) {
  const rateText = run.success != null ? percent(run.success) : run.landed != null ? `${percent(run.landed)}*` : '--';
  return el('tr', { class: selected ? 'selected' : '', onclick: () => onSelect(run.path) }, [
    el('td', { class: 'wrap' }, [run.name, el('div', { class: 'note', text: run.folder.slice(0, 15) })]),
    el('td', {}, el('span', { class: `state ${run.state}`, text: run.state })),
    el('td', { text: compact(run.timesteps) }),
    el('td', { text: fmt(run.fps, 0) }),
    el('td', { text: rateText }),
  ]);
}

function keyValues(entries) {
  const list = el('dl', { class: 'kv' });
  for (const [key, value] of entries) list.append(el('dt', { text: key }), el('dd', { text: value }));
  return list;
}

export function initTraining(options) {
  const ui = { list: byId('runList'), note: byId('runListNote'), detail: byId('runDetail'), file: byId('trainFile'),
               name: byId('trainName'), steps: byId('trainSteps'), start: byId('trainStart'), error: byId('trainError') };
  let selected = null;
  let timer = null;
  let detail = null;  // { path, charts, log, head }

  for (const t of options.trainings) {
    ui.file.append(el('option', { value: t.path, text: `${t.name} (${t.task})`, selected: t.path === DEFAULT_TRAINING }));
  }

  async function refresh() {
    try {
      const { runs, started_here: startedHere } = await getJson('/api/training/runs');
      if (!selected && runs.length) selected = runs[0].path;
      ui.list.replaceChildren(...runs.map((run) => runRow(run, run.path === selected, select)));
      const notes = ['* landed rate: solid rocket runs have no mission success rate.'];
      for (const p of startedHere) notes.push(`Started here: ${p.name} (${p.alive ? 'running' : `ended, code ${p.returncode}`}), log ${p.log}.`);
      ui.note.textContent = notes.join(' ');
      if (selected) await showRun(selected);
    } catch (error) {
      ui.note.textContent = `Could not read the runs: ${error.message}`;
    }
  }

  function select(path) {
    selected = path;
    refresh();
  }

  function buildDetail(path) {
    const head = el('div');
    const grid = el('div', { class: 'chart-grid' });
    const charts = {};
    for (const curve of CURVES) {
      const canvas = el('canvas');
      grid.append(el('div', { class: 'chart' }, [el('div', { class: 'title', text: curve.title }), canvas]));
      charts[curve.key] = new Chart(canvas, { yMin: curve.yMin ?? null, yMax: curve.yMax ?? null, minSpan: curve.minSpan ?? 0.1,
                                              xName: 'steps', xDigits: 0, digits: 3, emptyText: 'not in this run\'s log' });
    }
    const log = el('pre', { class: 'log' });
    ui.detail.replaceChildren(head, grid, el('h2', { text: 'Console log (end)' }), log);
    detail = { path, charts, log, head };
  }

  async function showRun(path) {
    const data = await getJson(`/api/training/run?path=${encodeURIComponent(path)}`);
    if (!detail || detail.path !== path) buildDetail(path);
    const run = data.run;
    const stop = run.stoppable ? el('button', { class: 'danger', text: 'Stop training', onclick: () => stopRun(run.name) }) : null;
    detail.head.replaceChildren(
      el('div', { class: 'run-head' }, [el('h3', { class: 'grow', text: run.folder }), el('span', { class: `state ${run.state}`, text: run.state }), stop]),
      keyValues([
        ['Task', run.task ?? '--'], ['Timesteps', compact(run.timesteps)], ['Speed', `${fmt(run.fps, 0)} steps/s`],
        ['Last update', ago(run.updated)], ['Policy file', run.has_policy ? 'policy.npz written' : 'not yet (written at the end)'],
        ['Summary', run.summary ? JSON.stringify(run.summary) : '--'],
      ]),
    );
    const x = data.series.timesteps;
    for (const curve of CURVES) {
      const chart = detail.charts[curve.key];
      chart.clear();
      let y = data.series[curve.key];
      if (curve.key === 'wind') y = y.map((v) => (v === YAML_WIND ? null : v));
      if (y.some((v) => v !== null)) chart.addSeries({ name: curve.key, x, y });
    }
    const atBottom = detail.log.scrollTop + detail.log.clientHeight >= detail.log.scrollHeight - 4;
    detail.log.textContent = data.log || `No console log found (looked for ${data.log_path ?? 'runs/train_logs/<name>.log'}).`;
    if (atBottom) detail.log.scrollTop = detail.log.scrollHeight;
  }

  async function stopRun(name) {
    try {
      await postJson('/api/training/stop', { name });
      refresh();
    } catch (error) {
      ui.error.textContent = error.message;
    }
  }

  async function start() {
    ui.error.textContent = '';
    const body = { training: ui.file.value, name: ui.name.value.trim(),
                   timesteps: ui.steps.value === '' ? null : Number(ui.steps.value) };
    try {
      const answer = await postJson('/api/training/start', body);
      ui.error.textContent = '';
      ui.note.textContent = `Started ${answer.name}; console log in ${answer.log}.`;
      selected = null;
      setTimeout(refresh, 3000);
    } catch (error) {
      ui.error.textContent = error.message;
    }
  }

  ui.start.addEventListener('click', start);
  return {
    shown() {
      refresh();
      clearInterval(timer);
      timer = setInterval(refresh, REFRESH_MS);
    },
    hidden() {
      clearInterval(timer);
    },
  };
}
