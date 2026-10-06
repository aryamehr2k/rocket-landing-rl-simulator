// The flight setup form of the Fly and Many flights tabs, built from the server's /api/options.

import { el, fmt } from './dom.js';

const SLIDERS = {
  wind_speed: { label: 'Wind speed', unit: 'm/s', min: 0, max: 10, step: 0.5, digits: 1 },
  wind_direction_deg: { label: 'Blowing toward', unit: 'deg', min: 0, max: 355, step: 5, digits: 0 },
  gust_std: { label: 'Gusts (std)', unit: 'm/s', min: 0, max: 3, step: 0.1, digits: 1 },
};
const ERRORS = {
  thrust_scale: { label: 'Thrust scale', step: 0.01 },
  dry_mass_offset_g: { label: 'Dry mass offset g', step: 10 },
  sensor_noise: { label: 'Sensor noise factor', step: 0.1 },
};
const OVERRIDES = {
  target_altitude: { label: 'Target height m', step: 1 },
  hover_time: { label: 'Hover time s', step: 1 },
  landing_radius: { label: 'Landing radius m', step: 0.5 },
  climb_speed: { label: 'Climb speed m/s', step: 0.1 },
  descent_speed: { label: 'Descent speed m/s', step: 0.1 },
};
const SPEED_NAMES = { max: 'As fast as possible' };
const DEFAULT_SPEED = '1';

function select(id, entries, chosen) {
  const node = el('select', { id });
  for (const { value, label } of entries) node.append(el('option', { value, text: label, selected: value === chosen }));
  return node;
}

function field(label, control, extra = null) {
  return el('label', { class: 'field' }, [el('span', {}, [label, extra]), control]);
}

function numberInput(id, attributes) {
  return el('input', { type: 'number', id, ...attributes });
}

function modelLabel(model) {
  return `${model.name} (${model.kind === 'run' ? 'run' : 'model'})`;
}

export function createFlightForm(form, options, { prefix, live }) {
  const id = (key) => `${prefix}-${key}`;
  const inputs = {};
  const listeners = [];
  let quiet = false;  // true while the page itself sets values, so no change is sent to the flight
  const defaults = options.defaults;
  const numbers = options.numbers;

  inputs.vehicle = select(id('vehicle'), options.vehicles.map((v) => ({ value: v.path, label: v.name })), defaults.vehicle);
  inputs.mission = select(id('mission'), options.missions.map((m) => ({ value: m.path, label: m.name })), defaults.mission);
  inputs.world = select(id('world'), options.worlds.map((w) => ({ value: w.path, label: w.name })), defaults.world);
  inputs.controller = select(id('controller'), [{ value: 'pid', label: 'PID' }].concat(
    options.models.map((m) => ({ value: m.path, label: modelLabel(m) }))), 'pid');
  inputs.seed = numberInput(id('seed'), { min: 0, step: 1, value: defaults.seed });
  form.append(el('fieldset', {}, [
    el('legend', { text: 'Flight' }),
    field('Vehicle', inputs.vehicle), field('Mission', inputs.mission), field('World (simulation settings)', inputs.world),
    el('div', { class: 'grid2' }, [field('Controller', inputs.controller), field(live ? 'Seed' : 'First seed', inputs.seed)]),
  ]));

  const windFields = Object.entries(SLIDERS).map(([key, spec]) => {
    inputs[key] = el('input', { type: 'range', id: id(key), min: spec.min, max: spec.max, step: spec.step, value: numbers[key].default });
    const shown = el('output', { for: id(key) });
    const update = () => { shown.textContent = fmt(Number(inputs[key].value), spec.digits, spec.unit); };
    inputs[key].addEventListener('input', () => { update(); emit('wind'); });
    update();
    return field(spec.label, inputs[key], shown);
  });
  form.append(el('fieldset', {}, [
    el('legend', { text: 'Wind' }),
    live ? el('p', { class: 'note', text: 'Changes apply at once during a flight. Direction is where the wind blows toward, from +x.' }) : null,
    ...windFields,
  ]));

  const errorFields = Object.entries(ERRORS).map(([key, spec]) => {
    const n = numbers[key];
    inputs[key] = numberInput(id(key), { min: n.min, max: n.max, step: spec.step, value: n.default });
    return field(spec.label, inputs[key]);
  });
  const errorBlock = [el('legend', { text: 'Hidden errors' }), el('div', { class: 'grid2' }, errorFields)];
  if (!live) {
    inputs.random_errors = el('input', { type: 'checkbox', id: id('random_errors') });
    errorBlock.push(el('label', { class: 'check' }, [inputs.random_errors, 'Draw thrust and mass errors per seed from the world file']));
  }
  form.append(el('fieldset', {}, errorBlock));

  const overrideFields = Object.entries(OVERRIDES).map(([key, spec]) => {
    const limits = options.overrides[key];
    inputs[key] = numberInput(id(key), { min: limits.min, max: limits.max, step: spec.step });
    return field(spec.label, inputs[key]);
  });
  form.append(el('fieldset', {}, [
    el('legend', { text: 'Mission overrides' }),
    el('p', { class: 'note', text: 'Empty fields keep the mission file value shown in grey.' }),
    el('div', { class: 'grid2' }, overrideFields),
  ]));

  if (live) {
    inputs.speed = select(id('speed'), options.speeds.map((s) => ({ value: String(s), label: SPEED_NAMES[s] || `${s}x real time` })), DEFAULT_SPEED);
    inputs.speed.addEventListener('change', () => emit('speed'));
    form.append(el('fieldset', {}, [el('legend', { text: 'Playback' }), field('Speed', inputs.speed)]));
  } else {
    inputs.flights = numberInput(id('flights'), { min: 1, max: options.max_flights, step: 1, value: 20 });
    inputs.compare_pid = el('input', { type: 'checkbox', id: id('compare'), checked: true });
    form.append(el('fieldset', {}, [
      el('legend', { text: 'Batch' }),
      field(`Number of flights (1 to ${options.max_flights})`, inputs.flights),
      el('label', { class: 'check' }, [inputs.compare_pid, 'Compare with the PID on the same seeds']),
      el('p', { class: 'note', text: `Runs on up to ${options.batch_processes} worker processes.` }),
    ]));
  }

  function emit(kind) {
    if (quiet) return;
    for (const listener of listeners) listener(kind);
  }

  function showMissionValues() {
    const mission = options.missions.find((m) => m.path === inputs.mission.value);
    for (const key of Object.keys(OVERRIDES)) inputs[key].placeholder = mission?.values ? String(mission.values[key]) : '';
  }

  function applyWorldWind() {
    const world = options.worlds.find((w) => w.path === inputs.world.value);
    if (!world?.wind) return;
    inputs.wind_speed.value = world.wind.speed;
    inputs.wind_direction_deg.value = world.wind.direction_deg;
    inputs.gust_std.value = world.wind.gust_std;
    for (const key of Object.keys(SLIDERS)) inputs[key].dispatchEvent(new Event('input'));
  }

  inputs.mission.addEventListener('change', showMissionValues);
  inputs.world.addEventListener('change', applyWorldWind);
  showMissionValues();

  function optionalNumber(input) {
    return input.value === '' ? null : Number(input.value);
  }

  return {
    values() {
      const body = {
        vehicle: inputs.vehicle.value, mission: inputs.mission.value, world: inputs.world.value,
        controller: inputs.controller.value, seed: Number(inputs.seed.value),
      };
      for (const key of [...Object.keys(SLIDERS), ...Object.keys(ERRORS)]) body[key] = Number(inputs[key].value);
      for (const key of Object.keys(OVERRIDES)) body[key] = optionalNumber(inputs[key]);
      if (live) body.speed = inputs.speed.value;
      else Object.assign(body, { flights: Number(inputs.flights.value), compare_pid: inputs.compare_pid.checked,
                                 random_errors: inputs.random_errors.checked });
      return body;
    },
    wind() {
      return { speed: Number(inputs.wind_speed.value), direction_deg: Number(inputs.wind_direction_deg.value),
               gust_std: Number(inputs.gust_std.value) };
    },
    speed() {
      return inputs.speed?.value;
    },
    selectController(path, label) {
      if (![...inputs.controller.options].some((o) => o.value === path)) {
        inputs.controller.append(el('option', { value: path, text: label || path }));
      }
      inputs.controller.value = path;
    },
    set(values, { quiet: silently = false } = {}) {
      quiet = silently;
      for (const [key, value] of Object.entries(values)) {
        if (!inputs[key]) continue;
        inputs[key].value = String(value);
        inputs[key].dispatchEvent(new Event('input'));
      }
      quiet = false;
    },
    onChange(listener) {
      listeners.push(listener);
    },
  };
}
