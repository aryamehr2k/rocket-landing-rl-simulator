// Entry point: loads the form options, sets up the four tabs and switches between them by the URL hash
// (#fly, #batch, #training, #models; #fly-demo also launches a PID flight).

import { getJson } from './api.js';
import { initBatch } from './batch.js';
import { byId } from './dom.js';
import { initFly } from './fly.js';
import { initModels } from './models.js';
import { initTraining } from './training.js';

const DEFAULT_TAB = 'fly';
const DEMO = 'demo';

function showTab(name) {
  for (const section of document.querySelectorAll('.tab')) section.classList.toggle('active', section.id === `tab-${name}`);
  for (const link of document.querySelectorAll('.tabs a')) link.classList.toggle('active', link.dataset.tab === name);
}

async function main() {
  const status = byId('serverStatus');
  let options;
  try {
    options = await getJson('/api/options');
    status.textContent = '';
  } catch (error) {
    status.textContent = `server not reachable: ${error.message}`;
    return;
  }
  const fly = initFly(options);
  const batch = initBatch(options);
  const training = initTraining(options);
  const models = initModels({
    onFly(path, label) {
      fly.selectModel(path, label);
      batch.selectModel(path, label);
      location.hash = '#fly';
    },
  });
  const tabs = { fly, batch, training, models };
  let current = null;

  function route() {
    const [name, option] = location.hash.slice(1).split('-');
    const tab = name in tabs ? name : DEFAULT_TAB;
    if (current !== tab) {
      tabs[current]?.hidden?.();
      current = tab;
      showTab(tab);
      tabs[tab].shown?.();
    }
    if (tab === 'fly' && option === DEMO) {
      history.replaceState(null, '', '#fly');
      fly.demo();
    }
  }

  window.addEventListener('hashchange', route);
  const demo = location.hash === `#fly-${DEMO}`;
  route();
  if (!demo) fly.attach();
  batch.restore();
}

main();
