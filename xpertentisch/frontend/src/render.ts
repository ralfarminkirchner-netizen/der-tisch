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
  const card = el('article', { class: 'card', 'data-job-id': job.id });
  card.append(
    el('header', {}, [
      el('h3', {}, [signet(job.label), job.label]),
      // Zwei Ebenen statt einer Reihe gleichrangiger Schilder: oben der
      // Zustand, darunter die Nebenangaben.
      el('div', { class: 'tags' }),
      el('div', { class: 'leiste' }),
    ]),
    el('div', { class: 'card-body' }),
    // Bleibt beim Aktualisieren stehen: hier hängen Aktionen und Kontextansicht.
    el('div', { class: 'card-foot' }),
  );
  updateCard(card, job, markers);
  return card;
}

/** Trägt die Herkunft, ohne eine Farbe zu belegen — Farbe gehört der Auswertung. */
function signet(label: string): HTMLElement {
  const zeichen = label
    .split(/[\s—–-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((teil) => teil[0])
    .join('')
    .toUpperCase();
  return el('span', { class: 'signet', 'aria-hidden': 'true' }, [zeichen || '?']);
}

export function updateCard(card: HTMLElement, job: Job, markers: Marker[]): void {
  // Die Lichtkante oben trägt den Zustand; die Beschriftung trägt ihn ebenso.
  card.className = `card zustand-${job.status}${
    card.classList.contains('highlight') ? ' highlight' : ''
  }`;
  const tags = card.querySelector('.tags');
  const leiste = card.querySelector('.leiste');
  const body = card.querySelector('.card-body');
  if (!tags || !leiste || !body) return;

  // ---- Zustand: das eine Schild, das zählt.
  tags.replaceChildren();
  tags.append(
    el('span', { class: `tag zustand ${job.status}` }, [STATUS_LABEL[job.status] ?? job.status]),
  );
  if (job.partial) tags.append(el('span', { class: 'tag partial' }, ['Teilantwort']));

  // ---- Nebenangaben: eine ruhige Zeile, nicht sechs gleiche Schilder.
  //
  // Getrennt wird durch Abstand und durch den Kontrast zwischen Benennung und
  // Wert — nicht durch Trennzeichen. Ein Trennpunkt stünde beim Zeilenumbruch
  // sonst als Rest am Zeilenanfang.
  leiste.replaceChildren();
  const angabe = (benennung: string, wert?: string, titel?: string, klasse = '') => {
    const span = el('span', titel ? { title: titel } : {});
    if (benennung) {
      span.append(el('span', { class: `leiste-name ${klasse}`.trim() }, [benennung]));
    }
    if (wert !== undefined) {
      span.append(el('span', { class: 'leiste-wert' }, [wert]));
    }
    leiste.append(span);
  };

  leiste.append(
    el('span', { class: 'leiste-herkunft' }, [`${job.provider} · ${job.model}`]),
  );
  if (job.latency_ms !== null && job.status !== 'running' && job.status !== 'streaming') {
    angabe('Dauer', formatDuration(job.latency_ms), 'gemessene Zeit bis zur Antwort');
  }
  if (job.tokens_in !== null || job.tokens_out !== null) {
    angabe(
      'Token',
      `${job.tokens_in ?? '?'}/${job.tokens_out ?? '?'}`,
      'vom Anbieter gemeldeter Verbrauch (ein/aus)',
    );
  }
  if (job.cost_source === 'berechnet' && job.cost_micro !== null) {
    angabe('Kosten', formatCost(job.cost_micro), 'mit den von dir eingetragenen Preisen berechnet');
  } else if ((job.tokens_in ?? job.tokens_out) !== null) {
    // Bleibt bewusst ein zusammenhängender Satzteil: ohne hinterlegte Preise
    // wird nichts geschätzt, und genau das soll dastehen.
    angabe('Kosten unbekannt', undefined, 'ohne hinterlegte Preise wird nichts geschätzt',
      'unbekannt');
  }

  // ---- Körper.
  body.replaceChildren();

  if (job.status === 'queued' || job.status === 'running') {
    // Ein Puls, kein Fortschrittsbalken: niemand kennt hier einen Fortschritt.
    body.append(
      el('div', { class: 'puls', role: 'presentation' }, [
        el('span', {}), el('span', {}), el('span', {}),
      ]),
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
