// Talking to the dashboard server: JSON requests and the live flight's event stream.

async function readAnswer(response) {
  const text = await response.text();
  let body = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
  }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

export async function getJson(url) {
  return readAnswer(await fetch(url, { cache: 'no-store' }));
}

export async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return readAnswer(response);
}

// Opens a Server-Sent Events stream; `handlers` maps event names to functions taking the parsed data.
export function openStream(url, handlers) {
  const source = new EventSource(url);
  for (const [name, handle] of Object.entries(handlers)) {
    if (name === 'error') source.addEventListener('error', handle);
    else source.addEventListener(name, (event) => handle(JSON.parse(event.data)));
  }
  return source;
}
