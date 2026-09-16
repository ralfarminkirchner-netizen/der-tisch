import { highlightAnswer } from './markers';
import type { Job, JobStatus, Marker, Summary, SparkEntry } from './types';

const STATUS_LABEL: Record<JobStatus, string> = {
  not_requested: 'nicht gefragt',
  queued: 'wartet',
  running: 'denkt nach',
  done: 'fertig',
  error: 'Fehler',
  interrupted: 'unterbrochen',
  cancelled: 'abgebrochen',
};

/** Warum eine Karte leer ist — die Fälle sind nicht dasselbe. */
const LEER_GRUND: Partial<Record<JobStatus, string>> = {
  not_requested: 'Für diesen Funken nicht angefragt.',
  done: 'Antwort kam an, enthielt aber keinen Text.',
  cancelled: 'Abgebrochen, bevor eine Antwort kam.',
};

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

function formatDuration(ms: number | null): string {
  if (!ms && ms !== 0) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
}

/**
 * Eine Modellkarte.
 *
 * Jede Karte lebt für sich: Zustand, Fehler und Text eines Modells wirken
 * sich nicht auf die Karten der anderen Modelle aus.
 */
export function createCard(job: Job, markers: Marker[]): HTMLElement {
  const card = el('article', { class: 'flaeche card', 'data-job-id': job.id });
  card.append(
    el('header', {}, [
      el('h3', {}, [job.label]),
      el('div', { class: 'tags' }),
    ]),
    el('div', { class: 'card-body' }),
    // Bleibt beim Aktualisieren stehen: hier hängen Aktionen und Kontextansicht.
    el('div', { class: 'card-foot' }),
  );
  updateCard(card, job, markers);
  return card;
}

export function updateCard(card: HTMLElement, job: Job, markers: Marker[]): void {
  // Die farbige Kante oben trägt den Zustand.
  card.className = `flaeche card zustand-${job.status}${
    card.classList.contains('highlight') ? ' highlight' : ''
  }`;
  const tags = card.querySelector('.tags');
  const body = card.querySelector('.card-body');
  if (!tags || !body) return;

  tags.replaceChildren();
  tags.append(el('span', { class: `tag ${job.status}` }, [STATUS_LABEL[job.status] ?? job.status]));
  if (job.partial) tags.append(el('span', { class: 'tag partial' }, ['Teilantwort']));
  tags.append(el('span', { class: 'tag' }, [`${job.provider} · ${job.model}`]));
  if (job.latency_ms !== null && job.status !== 'running') {
    tags.append(el('span', { class: 'tag' }, [formatDuration(job.latency_ms)]));
  }

  body.replaceChildren();

  if (job.status === 'queued' || job.status === 'running') {
    body.append(
      el('div', { class: 'skeleton' }),
      el('div', { class: 'skeleton' }),
      el('div', { class: 'skeleton' }),
    );
  }

  if (job.error) {
    body.append(el('p', { class: 'errorbox' }, [job.error]));
  }

  const text = job.text ?? '';
  if (text.length > 0) {
    const answer = el('div', { class: 'answer' });
    // Der Originaltext bleibt unverändert; Marker sind nur eine Auflage darüber.
    answer.innerHTML = highlightAnswer(text, markers.filter((m) => m.job_id === job.id));
    body.append(answer);
  } else if (LEER_GRUND[job.status]) {
    body.append(el('p', { class: 'hint' }, [LEER_GRUND[job.status]!]));
  }
}

export function renderTable(summary: Summary): HTMLElement {
  const panel = el('section', { class: 'flaeche panel' }, [el('h4', {}, ['Vergleich'])]);
  const wrap = el('div', { class: 'tablewrap' });
  const table = el('table');
  const head = el('tr');
  for (const label of [
    'Modell', 'Status', 'Zeichen', 'Sätze', 'Dauer', 'Übereinst.', 'Widerspr.', 'Einzigartig',
  ]) {
    head.append(el('th', {}, [label]));
  }
  table.append(el('thead', {}, [head]));

  const tbody = el('tbody');
  for (const row of summary.models) {
    const tr = el('tr', { 'data-job-id': row.job_id });
    tr.append(
      el('td', { class: 'name' }, [row.label]),
      el('td', {}, [STATUS_LABEL[row.status] ?? row.status]),
      el('td', {}, [String(row.chars)]),
      el('td', {}, [String(row.sentences)]),
      el('td', {}, [formatDuration(row.latency_ms)]),
      el('td', {}, [String(row.agreements)]),
      el('td', {}, [String(row.contradictions)]),
      el('td', {}, [String(row.unique)]),
    );
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  panel.append(wrap);
  panel.append(
    el('p', { class: 'hint' }, [
      'Die Tabelle beschreibt, sie bewertet nicht. Es entsteht keine Rangfolge der Modelle.',
    ]),
  );
  return panel;
}

export function renderQuestion(entry: SparkEntry): HTMLElement {
  return el('p', { class: 'question' }, [entry.spark.prompt]);
}

export function legend(): HTMLElement {
  return el('div', { class: 'legend' }, [
    el('span', { class: 'l-agree' }, ['Übereinstimmung']),
    el('span', { class: 'l-contra' }, ['Widerspruch']),
    el('span', { class: 'l-unique' }, ['Einzigartig']),
  ]);
}
