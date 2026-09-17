import { highlightAnswer } from './markers';
import type { Job, JobStatus, Marker, Summary, SparkEntry } from './types';

const STATUS_LABEL: Record<JobStatus, string> = {
  not_requested: 'nicht gefragt',
  queued: 'wartet',
  running: 'denkt nach',
  streaming: 'schreibt',
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

/** Kosten aus Millionstel der eingetragenen Währung. */
function formatCost(micro: number): string {
  const betrag = micro / 1_000_000;
  return betrag < 0.01 ? `${(betrag * 100).toFixed(2)} ct` : betrag.toFixed(4);
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
  card.style.viewTransitionName = `karte-${cssIdent(job.id)}`;
  card.append(
    el('header', {}, [
      el('div', { class: 'card-titel' }, [
        el('h3', {}, [job.label]),
        el('p', { class: 'card-herkunft' }, []),
      ]),
      el('div', { class: 'tags' }),
    ]),
    el('div', { class: 'card-body' }),
    // Bleibt beim Aktualisieren stehen: hier hängen Aktionen und Kontextansicht.
    el('div', { class: 'card-foot' }),
  );
  updateCard(card, job, markers);
  return card;
}

function cssIdent(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-');
}

export function updateCard(card: HTMLElement, job: Job, markers: Marker[]): void {
  card.className = `flaeche card zustand-${job.status}${
    card.classList.contains('highlight') ? ' highlight' : ''
  }`;
  const tags = card.querySelector('.tags');
  const body = card.querySelector('.card-body');
  const herkunft = card.querySelector('.card-herkunft');
  if (!tags || !body) return;

  if (herkunft) herkunft.textContent = `${job.provider} · ${job.model}`;

  tags.replaceChildren();
  tags.append(el('span', { class: `tag ${job.status}` }, [STATUS_LABEL[job.status] ?? job.status]));
  if (job.partial) tags.append(el('span', { class: 'tag partial' }, ['Teilantwort']));
  if (job.latency_ms !== null && job.status !== 'running' && job.status !== 'streaming') {
    tags.append(el('span', { class: 'tag meta' }, [formatDuration(job.latency_ms)]));
  }
  if (job.tokens_in !== null || job.tokens_out !== null) {
    tags.append(
      el('span', { class: 'tag meta', title: 'verbrauchte Token (ein/aus)' }, [
        `${job.tokens_in ?? '?'}/${job.tokens_out ?? '?'} Token`,
      ]),
    );
  }
  if (job.cost_source === 'berechnet' && job.cost_micro !== null) {
    tags.append(
      el('span', { class: 'tag meta', title: 'mit den von dir eingetragenen Preisen berechnet' }, [
        formatCost(job.cost_micro),
      ]),
    );
  } else if ((job.tokens_in ?? job.tokens_out) !== null) {
    tags.append(
      el('span', { class: 'tag meta', title: 'ohne hinterlegte Preise wird nichts geschätzt' }, [
        'Kosten unbekannt',
      ]),
    );
  }

  body.replaceChildren();

  if (job.status === 'queued' || job.status === 'running') {
    const vorgang = el('div', {
      class: 'satzvorgang',
      'aria-hidden': 'true',
    });
    vorgang.append(el('span'), el('span'), el('span'));
    body.append(vorgang);
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
