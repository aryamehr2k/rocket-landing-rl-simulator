// The Fly tab: launch a live flight, follow it through the event stream and steer the wind and pushes.

import { getJson, openStream, postJson } from './api.js';
import { byId, el } from './dom.js';
import { createFlightForm } from './flightform.js';
import { FlyPanel } from './flypanel.js';

const WIND_DEBOUNCE_MS = 120;
const DEMO_SETTINGS = { wind_speed: 3, wind_direction_deg: 45, gust_std: 0.5, speed: '2' };

export function initFly(options) {
  const form = createFlightForm(byId('flyForm'), options, { prefix: 'fly', live: true });
  const panel = new FlyPanel();
  const ui = {
    launch: byId('flyLaunch'), pause: byId('flyPause'), stop: byId('flyStop'), error: byId('flyError'),
    phase: byId('hudPhase'), time: byId('hudTime'), controller: byId('hudController'), rate: byId('hudRate'),
    follow: byId('followCamera'), charts: byId('showCharts'), pushes: document.querySelectorAll('[data-push]'),
  };
  let scene = null;
  let source = null;
  let state = 'idle';
  let windTimer = null;
  let flightInfo = null;

  import('./scene.js')
    .then(({ FlightScene }) => {
      scene = new FlightScene(byId('flyScene'));
      scene.setFollow(ui.follow.checked);
      scene.setup(flightInfo ?? preview());
    })
    .catch((error) => {
      byId('flyScene').append(el('div', {
        class: 'scene-message', text: `3D view unavailable: ${error.message}. Everything else still works.`,
      }));
    });

  // The vehicle on the pad and the mission markers before the first launch.
  function preview() {
    const vehicle = options.vehicles.find((v) => v.path === options.defaults.vehicle) || options.vehicles[0];
    const mission = options.missions.find((m) => m.path === options.defaults.mission) || options.missions[0];
    return { vehicle: vehicle.geometry, mission: { ...mission.values, altitude_tolerance: mission.altitude_tolerance } };
  }

  function setState(next, paused = false) {
    state = next;
    const flying = next === 'running';
    ui.pause.disabled = !flying;
    ui.stop.disabled = !flying;
    ui.pause.textContent = paused ? 'Resume' : 'Pause';
    for (const button of ui.pushes) button.disabled = !flying || paused;
  }

  function showError(error) {
    ui.error.textContent = error ? error.message || String(error) : '';
  }

  async function launch() {
    showError(null);
    ui.launch.disabled = true;
    try {
      await postJson('/api/fly/start', form.values());
      connect();
    } catch (error) {
      showError(error);
    } finally {
      ui.launch.disabled = false;
    }
  }

  function connect() {
    source?.close();
    source = openStream('/api/fly/stream', {
      setup: onSetup,
      frame: onFrame,
      result: onResult,
      idle: () => source.close(),
    });
  }

  function onSetup(info) {
    flightInfo = info;
    const wind = info.wind;
    form.set({ wind_speed: wind.speed, wind_direction_deg: wind.direction_deg, gust_std: wind.gust_std, speed: info.speed ?? 'max' },
             { quiet: true });
    panel.reset(info);
    scene?.setup(info);
    ui.controller.textContent = `${info.controller}, seed ${info.seed}`;
    setState('running');
  }

  function onFrame(frame) {
    panel.frame(frame);
    scene?.update(frame);
    ui.phase.textContent = frame.phase;
    ui.time.textContent = `t ${frame.t.toFixed(1)} s`;
    ui.rate.textContent = Number.isFinite(frame.rate) ? `${frame.rate.toFixed(1)}x` : '';
  }

  function onResult(result) {
    source.close();
    panel.result(result);
    setState('finished');
    ui.rate.textContent = result.success ? 'mission OK' : 'mission failed';
  }

  async function control(body) {
    try {
      const answer = await postJson('/api/fly/control', body);
      setState(answer.state === 'finished' ? 'finished' : 'running', answer.paused);
      if (answer.paused) ui.rate.textContent = 'paused';
    } catch (error) {
      showError(error);
    }
  }

  form.onChange((kind) => {
    if (state !== 'running') return;
    if (kind === 'speed') control({ speed: form.speed() });
    if (kind === 'wind') {
      clearTimeout(windTimer);
      windTimer = setTimeout(() => control({ wind: form.wind() }), WIND_DEBOUNCE_MS);
    }
  });
  ui.launch.addEventListener('click', launch);
  ui.pause.addEventListener('click', () => control({ paused: ui.pause.textContent === 'Pause' }));
  ui.stop.addEventListener('click', () => postJson('/api/fly/stop').catch(showError));
  for (const button of ui.pushes) button.addEventListener('click', () => control({ push: button.dataset.push }));
  ui.follow.addEventListener('change', () => scene?.setFollow(ui.follow.checked));
  ui.charts.addEventListener('change', () => {
    byId('flyChartsBlock').classList.toggle('hidden', !ui.charts.checked);
    byId('tab-fly').classList.toggle('no-charts', !ui.charts.checked);
    panel.redraw();
  });

  return {
    // Show a flight that is already running or finished (after a page reload).
    async attach() {
      try {
        const status = await getJson('/api/fly/status');
        if (status.state !== 'idle') connect();
      } catch (error) {
        showError(error);
      }
    },
    // Fly the PID with some wind at twice real time, for a first look and for screenshots.
    demo() {
      form.set({ ...DEMO_SETTINGS, controller: 'pid' });
      launch();
    },
    selectModel(path, label) {
      form.selectController(path, label);
    },
    shown() {
      scene?.resize();
      panel.redraw();
    },
  };
}
