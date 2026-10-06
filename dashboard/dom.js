// Small helpers for building the page and formatting numbers.

export function el(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

export function fmt(value, digits = 1, unit = '') {
  if (value === null || value === undefined || !Number.isFinite(value)) return '--';
  return unit ? `${value.toFixed(digits)} ${unit}` : value.toFixed(digits);
}

export function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '--';
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function byId(id) {
  return document.getElementById(id);
}

// Short thousands form for step counts: 491520 -> "492k".
export function compact(value) {
  if (!Number.isFinite(value)) return '--';
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${Math.round(value / 1e3)}k`;
  return String(Math.round(value));
}

export function ago(seconds) {
  if (!Number.isFinite(seconds)) return '--';
  const elapsed = Date.now() / 1000 - seconds;
  if (elapsed < 90) return `${Math.max(0, Math.round(elapsed))} s ago`;
  if (elapsed < 5400) return `${Math.round(elapsed / 60)} min ago`;
  if (elapsed < 172800) return `${Math.round(elapsed / 3600)} h ago`;
  return `${Math.round(elapsed / 86400)} days ago`;
}
