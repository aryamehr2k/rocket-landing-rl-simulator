// The Models tab: exported models and finished hop runs, what goes in and out of each network, and a
// button that hands the model to the Fly tab.

import { getJson } from './api.js';
import { ago, byId, el, fmt } from './dom.js';

const SIGNIFICANT = 4;
const OVERVIEW = [
  ['Task', (i) => i.task], ['Source run', (i) => i.source_run], ['Vehicle', (i) => i.vehicle], ['Mission', (i) => i.mission],
  ['Network', (i) => i.network && `${i.network.layer_sizes.join(' - ')} (${i.network.activation}, ${i.network.parameters} parameters)`],
  ['Control rate', (i) => i.control_rate_hz && `${i.control_rate_hz} Hz, action delay ${i.action_delay_steps} steps`],
  ['Controllers', (i) => i.controllers && `steering ${i.controllers.steering}, throttle ${i.controllers.throttle}, mode ${i.controllers.mode}`],
  ['Hover throttle', (i) => Number.isFinite(i.hover_throttle_nominal) && fmt(i.hover_throttle_nominal, 3)],
  ['Max gimbal', (i) => Number.isFinite(i.max_gimbal_deg) && `${fmt(i.max_gimbal_deg, 1)} deg`],
  ['Input clip', (i) => i.input_clip], ['Input rule', (i) => i.input_rule], ['Created', (i) => i.created],
];
const INPUT_COLUMNS = ['index', 'name', 'scale', 'mean', 'std', 'meaning'];
const OUTPUT_COLUMNS = ['index', 'name', 'range', 'meaning'];

function text(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(+value.toPrecision(SIGNIFICANT));
  if (Array.isArray(value) && value.every((v) => typeof v !== 'object')) return value.map(text).join(', ');
  if (value === null || value === undefined) return '--';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function table(rows, columns) {
  const isText = (column) => rows.some((row) => typeof row[column] === 'string');
  const kind = (column) => (column === 'meaning' ? 'wrap' : isText(column) ? 'text' : null);
  return el('table', {}, [
    el('thead', {}, el('tr', {}, columns.map((c) => el('th', { class: kind(c), text: c })))),
    el('tbody', {}, rows.map((row) => el('tr', {}, columns.map((c) => el('td', { class: kind(c), text: text(row[c]) }))))),
  ]);
}

// Any JSON value as nested key and value lists, with lists of objects as tables.
function anyValue(value) {
  if (Array.isArray(value) && value.length && value.every((v) => v && typeof v === 'object' && !Array.isArray(v))) {
    return table(value, [...new Set(value.flatMap((v) => Object.keys(v)))]);
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const list = el('dl', { class: 'kv' });
    for (const [key, item] of Object.entries(value)) {
      const nested = item && typeof item === 'object' && !(Array.isArray(item) && item.every((v) => typeof v !== 'object'));
      list.append(el('dt', { text: key }), el('dd', {}, nested ? anyValue(item) : text(item)));
    }
    return list;
  }
  return el('span', { text: text(value) });
}

function section(title, content) {
  return [el('h2', { text: title }), content];
}

export function initModels({ onFly }) {
  const ui = { list: byId('modelList'), detail: byId('modelDetail') };
  let models = [];
  let selected = null;

  function card(model) {
    return el('button', { type: 'button', class: `card${model.path === selected ? ' selected' : ''}`, onclick: () => show(model.path) }, [
      el('div', { text: model.name }),
      el('div', { class: 'kind', text: `${model.kind === 'run' ? 'training run' : 'exported model'}, ${ago(model.created)}` }),
    ]);
  }

  function show(path) {
    selected = path;
    ui.list.replaceChildren(...models.map(card));
    const model = models.find((m) => m.path === path);
    if (!model) return;
    const label = `${model.name} (${model.kind})`;
    const blocks = [
      el('div', { class: 'run-head' }, [
        el('h3', { class: 'grow', text: model.name }),
        el('button', { class: 'primary', type: 'button', text: 'Fly this model', onclick: () => onFly(model.path, label) }),
      ]),
      el('dl', { class: 'kv' }, [el('dt', { text: 'Folder' }), el('dd', { text: model.path }),
                                  el('dt', { text: 'Training file' }), el('dd', { text: model.training ?? '--' })]),
    ];
    const info = model.info;
    if (info) {
      const overview = el('dl', { class: 'kv' });
      for (const [key, pick] of OVERVIEW) {
        const value = pick(info);
        if (value !== undefined && value !== null && value !== false) overview.append(el('dt', { text: key }), el('dd', { text: text(value) }));
      }
      blocks.push(...section('Overview', overview));
      if (info.inputs?.length) blocks.push(...section('Inputs (once per plane)', table(info.inputs, INPUT_COLUMNS)));
      if (info.outputs?.length) blocks.push(...section('Outputs', table(info.outputs, OUTPUT_COLUMNS)));
      if (info.evaluation) blocks.push(...section('Evaluation', anyValue(info.evaluation)));
      if (info.checks) blocks.push(...section('Export checks', anyValue(info.checks)));
    } else {
      blocks.push(el('p', { class: 'note', text: 'No model.json here. scripts/export_policy.py writes one with the inputs, outputs and checks.' }));
    }
    const summary = model.summary ?? info?.training;
    if (summary) blocks.push(...section('Training summary', anyValue(summary)));
    ui.detail.replaceChildren(...blocks);
  }

  async function refresh() {
    try {
      models = (await getJson('/api/models')).models;
    } catch (error) {
      ui.detail.replaceChildren(el('p', { class: 'error', text: error.message }));
      return;
    }
    if (!models.length) {
      ui.list.replaceChildren(el('p', { class: 'muted', text: 'No models yet. A hop training run appears here when it finishes.' }));
      return;
    }
    show(models.some((m) => m.path === selected) ? selected : models[0].path);
  }

  return { shown: refresh };
}
