import './styles.css';

import { ApiError, api, connectEvents, newRequestId } from './api';
import { renderGraph, type GraphSelection } from './graph';
import { createCard, el, legend, renderQuestion, renderTable, updateCard } from './render';
import { renderSettings } from './settings';
import type {
  AppConfig,
  HealthInfo,
  Job,
  Marker,
  SessionBundle,
  SparkEntry,
  Summary,
} from './types';

const STORAGE_KEY = 'xpertentisch.session';

interface AppState {
  config: AppConfig | null;
  health: HealthInfo | null;
  bundle: SessionBundle | null;
  selectedModels: Set<string>;
  streamOpen: boolean;
  settingsOpen: boolean;
  /** Kennung des laufenden Absendevorgangs — bleibt bei Wiederholung gleich. */
  pendingRequestId: string | null;
}

const state: AppState = {
  config: null,
  health: null,
  bundle: null,
  selectedModels: new Set(),
  streamOpen: false,
  settingsOpen: false,
  pendingRequestId: null,
};

let disconnect: (() => void) | null = null;
const blocks = new Map<string, HTMLElement>();
/** Das Einstellungsfeld überlebt Neuzeichnungen — sonst wären Eingaben weg. */
let settingsPanel: HTMLElement | null = null;

const root = document.getElementById('app')!;

// --------------------------------------------------------------------- Start

async function boot(): Promise<void> {
  try {
    const [config, health] = await Promise.all([api.config(), api.health()]);
    state.config = config;
    state.health = health;
    state.selectedModels = new Set(config.models.map((m) => m.id));
  } catch (error) {
    root.replaceChildren(
      el('div', { class: 'glass' }, [
        el('h1', {}, ['XPERTENTiSCH']),
        el('p', { class: 'hint error' }, [
          `Der Dienst ist nicht erreichbar: ${(error as Error).message}`,
        ]),
      ]),
    );
    return;
  }

  const sessionId = sessionIdFromLocation() ?? localStorage.getItem(STORAGE_KEY);
  if (sessionId) {
    try {
      state.bundle = await api.getSession(sessionId);
    } catch {
      state.bundle = null;
    }
  }
  if (!state.bundle) {
    const session = await api.createSession(defaultTitle());
    state.bundle = { session, sparks: [], last_event_id: 0, exported_at: Date.now() / 1000 };
  }
  rememberSession(state.bundle.session.id);
  renderShell();
  openStream();
}

function sessionIdFromLocation(): string | null {
  const match = window.location.hash.match(/#\/s\/([A-Za-z0-9_]+)/);
  return match ? match[1] : null;
}

function rememberSession(id: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* Privater Modus ohne Speicher — die Sitzung lebt dann nur im Verlauf. */
  }
  const target = `#/s/${id}`;
  if (window.location.hash !== target) window.history.replaceState(null, '', target);
}

function defaultTitle(): string {
  const now = new Date();
  return `Tischrunde ${now.toLocaleDateString('de-DE')} ${now.toLocaleTimeString('de-DE', {
    hour: '2-digit',
    minute: '2-digit',
  })}`;
}

// -------------------------------------------------------------------- Gerüst

function renderShell(): void {
  const bundle = state.bundle!;
  root.removeAttribute('aria-busy');
  root.replaceChildren();

  root.append(header());

  if (state.settingsOpen) {
    if (!settingsPanel) {
      settingsPanel = renderSettings(Boolean(state.config?.settings_available), () => {
        state.settingsOpen = false;
        settingsPanel = null;
        renderShell();
        void refreshHealth();
      });
    }
    root.append(settingsPanel);
  } else {
    settingsPanel = null;
  }

  if (state.config?.fake_providers_enabled) {
    root.append(
      el('div', { class: 'glass testbanner' }, [
        'Achtung: Test-Provider sind aktiv. Die Antworten stammen nicht von echten Modellen.',
      ]),
    );
  }
  root.append(providerWarnings());
  root.append(sparkForm());

  const stream = el('div', { id: 'sparks' });
  root.append(stream);
  blocks.clear();
  for (const entry of bundle.sparks) {
    const block = renderSparkBlock(entry);
    blocks.set(entry.spark.id, block);
    stream.append(block);
  }

  root.append(closingSection());
  root.append(
    el('footer', { class: 'foot' }, [
      'XPERTENTiSCH — mehrere Modelle antworten unabhängig auf denselben Funken. ' +
        'Keine Rangfolge, keine verbindliche Lehre; die Einordnung bleibt bei dir.',
    ]),
  );
}

async function refreshHealth(): Promise<void> {
  try {
    const [health, config] = await Promise.all([api.health(), api.config()]);
    state.health = health;
    state.config = config;
    renderShell();
  } catch {
    /* Der bisherige Zustand bleibt stehen. */
  }
}

function header(): HTMLElement {
  const bundle = state.bundle!;
  const title = el('h1', {}, []);
  title.append(document.createTextNode('XPERTEN'), el('span', {}, ['TiSCH']));

  const status = el('div', { class: 'statusline', id: 'statusline' });
  updateStatusline(status);

  const zahnrad = el('button', {
    type: 'button',
    class: 'iconbutton',
    id: 'settings-open',
    'aria-label': 'Einstellungen öffnen',
    title: 'Einstellungen — Zugangsdaten und Modelle',
  }, ['⚙︎']);
  zahnrad.addEventListener('click', () => {
    state.settingsOpen = !state.settingsOpen;
    renderShell();
    if (state.settingsOpen) {
      document.getElementById('einstellungen')?.scrollIntoView({ block: 'nearest' });
    }
  });

  return el('header', { class: 'top' }, [
    el('div', { class: 'headline' }, [
      title,
      el('p', { class: 'sub' }, [bundle.session.title]),
    ]),
    el('div', { class: 'row' }, [status, zahnrad]),
  ]);
}

function updateStatusline(node?: HTMLElement | null): void {
  const line = node ?? document.getElementById('statusline');
  if (!line) return;
  const bundle = state.bundle!;
  line.replaceChildren();
  line.append(
    el('span', { class: `dot ${state.streamOpen ? 'ok' : 'warn'}` }),
    el('span', {}, [state.streamOpen ? 'Verbunden' : 'Verbindung wird aufgebaut …']),
    el('span', {}, ['·']),
    el('span', {}, [
      bundle.session.status === 'offen' ? 'Sitzung offen' : 'Sitzung abgeschlossen',
    ]),
  );
}

function providerWarnings(): HTMLElement {
  const health = state.health;
  const container = el('div', {});
  if (!health) return container;
  const broken = Object.entries(health.providers).filter(([, info]) => !info.ready);
  if (broken.length === 0) return container;
  const banner = el('div', { class: 'glass testbanner' }, [
    el('p', { class: 'hint' }, [
      `Nicht einsatzbereit: ${broken
        .map(([name, info]) => `${name} (${info.reason})`)
        .join(', ')}. Die übrigen Modelle antworten trotzdem.`,
    ]),
  ]);
  if (!state.settingsOpen) {
    const hin = el('button', { type: 'button' }, ['Zu den Einstellungen']);
    hin.addEventListener('click', () => {
      state.settingsOpen = true;
      renderShell();
      document.getElementById('einstellungen')?.scrollIntoView({ block: 'nearest' });
    });
    banner.append(hin);
  }
  container.append(banner);
  return container;
}

// ------------------------------------------------------------------- Eingabe

function sparkForm(): HTMLElement {
  const config = state.config!;
  const bundle = state.bundle!;
  const closed = bundle.session.status === 'abgeschlossen';

  const textarea = el('textarea', {
    id: 'prompt',
    placeholder: 'Funke — die Frage, die alle Modelle unabhängig beantworten sollen …',
    maxlength: String(config.max_prompt_chars),
    'aria-label': 'Funke',
  });

  const picker = el('div', { class: 'modelpicker' });
  for (const model of config.models) {
    const box = el('input', { type: 'checkbox', value: model.id });
    (box as HTMLInputElement).checked = state.selectedModels.has(model.id);
    box.addEventListener('change', () => {
      if ((box as HTMLInputElement).checked) state.selectedModels.add(model.id);
      else state.selectedModels.delete(model.id);
    });
    picker.append(el('label', {}, [box, `${model.label}`]));
  }

  const send = el('button', { class: 'primary', type: 'submit' }, ['Funke setzen']);
  const message = el('p', { class: 'hint', id: 'spark-message' }, []);

  const form = el('form', { class: 'glass spark' }, [
    textarea,
    el('div', { class: 'row spread' }, [picker, send]),
    message,
  ]);

  if (closed) {
    (textarea as HTMLTextAreaElement).disabled = true;
    (send as HTMLButtonElement).disabled = true;
    message.textContent = 'Die Sitzung ist abgeschlossen. Neue Funken sind nicht mehr möglich.';
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const prompt = (textarea as HTMLTextAreaElement).value.trim();
    message.className = 'hint';
    if (!prompt) {
      message.textContent = 'Bitte zuerst einen Funken eingeben.';
      return;
    }
    if (state.selectedModels.size === 0) {
      message.textContent = 'Bitte mindestens ein Modell auswählen.';
      return;
    }
    // Dieselbe Kennung für alle Wiederholungen dieses Absendevorgangs.
    state.pendingRequestId = state.pendingRequestId ?? newRequestId();
    (send as HTMLButtonElement).disabled = true;
    message.textContent = 'Der Funke läuft an …';
    try {
      const result = await api.createSpark(
        bundle.session.id,
        prompt,
        state.pendingRequestId,
        [...state.selectedModels],
      );
      state.pendingRequestId = null;
      (textarea as HTMLTextAreaElement).value = '';
      message.textContent = result.duplicate
        ? 'Dieser Funke lief bereits — es wurden keine neuen Aufträge gestartet.'
        : '';
      ensureSpark({
        spark: result.spark,
        jobs: result.jobs,
        markers: [],
        summary: null,
      });
    } catch (error) {
      message.className = 'hint error';
      const detail = error instanceof ApiError ? error.message : (error as Error).message;
      message.textContent = `Der Funke konnte nicht gesetzt werden: ${detail}. ` +
        'Erneutes Senden wiederholt denselben Auftrag, ohne ihn zu verdoppeln.';
    } finally {
      (send as HTMLButtonElement).disabled = bundle.session.status !== 'offen';
    }
  });

  return form;
}

// -------------------------------------------------------------------- Funken

function renderSparkBlock(entry: SparkEntry): HTMLElement {
  const block = el('section', { class: 'spark-block', 'data-spark-id': entry.spark.id }, [
    el('h2', {}, [`Funke ${entry.spark.seq}`]),
    renderQuestion(entry),
  ]);

  const cards = el('div', { class: 'cards' });
  for (const job of entry.jobs) {
    cards.append(createCard(job, entry.markers));
  }
  block.append(cards);

  const panels = el('div', { class: 'panel-grid' });
  block.append(panels);
  if (entry.summary) renderPanels(block, entry.summary);
  return block;
}

function renderPanels(block: HTMLElement, summary: Summary): void {
  const panels = block.querySelector('.panel-grid');
  if (!panels) return;
  panels.replaceChildren();
  panels.append(renderTable(summary));

  const graphPanel = el('section', { class: 'glass panel' }, [el('h4', {}, ['Beziehungsnetz'])]);
  const wrap = el('div', { class: 'graphwrap' });
  wrap.append(renderGraph(summary, (selection) => openAnswers(block, selection)));
  graphPanel.append(wrap, legend());
  graphPanel.append(
    el('p', { class: 'hint' }, [
      'Knoten oder Kante anklicken, um die zugehörigen Antworten zu öffnen.',
    ]),
  );
  panels.append(graphPanel);
}

/** Öffnet genau die Antworten, die zum angeklickten Graphelement gehören. */
function openAnswers(block: HTMLElement, selection: GraphSelection): void {
  const cards = block.querySelectorAll<HTMLElement>('.card');
  cards.forEach((card) => card.classList.remove('highlight'));
  let first: HTMLElement | null = null;
  for (const jobId of selection.jobIds) {
    const card = block.querySelector<HTMLElement>(`.card[data-job-id="${cssEscape(jobId)}"]`);
    if (!card) continue;
    card.classList.add('highlight');
    if (!first) first = card;
  }
  first?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  const message = block.querySelector<HTMLElement>('.selection-note');
  const note = message ?? el('p', { class: 'hint selection-note' }, []);
  note.textContent = `Geöffnet: ${selection.label}`;
  if (!message) block.querySelector('.panel-grid')?.after(note);
}

function cssEscape(value: string): string {
  return value.replace(/["\\]/g, '\\$&');
}

function ensureSpark(entry: SparkEntry): HTMLElement {
  const bundle = state.bundle!;
  const existing = blocks.get(entry.spark.id);
  if (existing) return existing;
  bundle.sparks.push(entry);
  const block = renderSparkBlock(entry);
  blocks.set(entry.spark.id, block);
  document.getElementById('sparks')?.append(block);
  return block;
}

function findEntry(predicate: (entry: SparkEntry) => boolean): SparkEntry | undefined {
  return state.bundle?.sparks.find(predicate);
}

// ---------------------------------------------------------------- Ereignisse

function openStream(): void {
  const bundle = state.bundle!;
  disconnect?.();
  disconnect = connectEvents(bundle.session.id, bundle.last_event_id, {
    onOpen: () => {
      state.streamOpen = true;
      updateStatusline();
    },
    onError: () => {
      state.streamOpen = false;
      updateStatusline();
    },
    onEvent: (type, payload) => handleEvent(type, payload as Record<string, unknown>),
  });
}

function handleEvent(type: string, payload: Record<string, unknown>): void {
  const bundle = state.bundle;
  if (!bundle) return;

  switch (type) {
    case 'funke.angelegt': {
      const spark = payload.spark as SparkEntry['spark'];
      const jobs = payload.jobs as Job[];
      ensureSpark({ spark, jobs, markers: [], summary: null });
      break;
    }
    case 'auftrag.laeuft':
      patchJob(String(payload.job_id), (job) => ({ ...job, status: 'running' }));
      break;
    case 'auftrag.fertig':
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'done',
        text: String(payload.text ?? ''),
        partial: Boolean(payload.partial),
        latency_ms: (payload.latency_ms as number) ?? null,
        error: null,
      }));
      break;
    case 'auftrag.fehler':
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'error',
        text: String(payload.text ?? ''),
        partial: Boolean(payload.partial),
        error: String(payload.error ?? 'Unbekannter Fehler'),
        latency_ms: (payload.latency_ms as number) ?? null,
      }));
      break;
    case 'auftrag.unterbrochen':
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'interrupted',
        error: String(payload.error ?? 'Unterbrochen'),
      }));
      break;
    case 'einschaetzung.fertig': {
      const sparkId = String(payload.spark_id);
      const entry = findEntry((e) => e.spark.id === sparkId);
      if (!entry) break;
      entry.summary = payload.summary as Summary;
      entry.markers = (payload.markers as Marker[]) ?? [];
      const block = blocks.get(sparkId);
      if (block) {
        for (const job of entry.jobs) {
          const card = block.querySelector<HTMLElement>(
            `.card[data-job-id="${cssEscape(job.id)}"]`,
          );
          if (card) updateCard(card, job, entry.markers);
        }
        renderPanels(block, entry.summary);
      }
      break;
    }
    case 'sitzung.abgeschlossen': {
      bundle.session = payload.session as SessionBundle['session'];
      renderShell();
      break;
    }
    default:
      break;
  }
}

/** Aktualisiert genau eine Modellkarte — die übrigen bleiben unberührt. */
function patchJob(jobId: string, update: (job: Job) => Job): void {
  const entry = findEntry((e) => e.jobs.some((j) => j.id === jobId));
  if (!entry) return;
  const index = entry.jobs.findIndex((j) => j.id === jobId);
  entry.jobs[index] = update(entry.jobs[index]);
  const block = blocks.get(entry.spark.id);
  const card = block?.querySelector<HTMLElement>(`.card[data-job-id="${cssEscape(jobId)}"]`);
  if (card) updateCard(card, entry.jobs[index], entry.markers);
}

// ------------------------------------------------------------------ Abschluss

function closingSection(): HTMLElement {
  const bundle = state.bundle!;
  const closed = bundle.session.status === 'abgeschlossen';
  const base = `/api/sessions/${bundle.session.id}`;

  const exports = el('div', { class: 'row' }, [
    el('a', { class: 'btn', href: `${base}/report.html`, download: '' }, ['Bericht als HTML']),
    el('a', { class: 'btn', href: `${base}/report.md`, download: '' }, ['Bericht als Markdown']),
  ]);

  if (closed) {
    const section = el('section', { class: 'glass closing closed-banner' }, [
      el('h4', {}, ['Sitzung abgeschlossen']),
    ]);
    if (bundle.session.closing_note) {
      section.append(el('p', { class: 'question' }, [bundle.session.closing_note]));
    }
    section.append(exports);
    return section;
  }

  const note = el('textarea', {
    id: 'closing-note',
    placeholder: 'Abschlussnotiz — was nimmst du mit? (freiwillig)',
    'aria-label': 'Abschlussnotiz',
  });
  const button = el('button', { type: 'button' }, ['Sitzung abschließen']);
  const hint = el('p', { class: 'hint' }, [
    'Nach dem Abschluss bleiben alle Antworten unverändert lesbar; neue Funken sind dann nicht mehr möglich.',
  ]);

  button.addEventListener('click', async () => {
    (button as HTMLButtonElement).disabled = true;
    try {
      const session = await api.closeSession(
        bundle.session.id,
        (note as HTMLTextAreaElement).value.trim(),
      );
      bundle.session = session;
      renderShell();
    } catch (error) {
      hint.className = 'hint error';
      hint.textContent = `Abschluss fehlgeschlagen: ${(error as Error).message}`;
      (button as HTMLButtonElement).disabled = false;
    }
  });

  return el('section', { class: 'glass closing' }, [
    el('h4', {}, ['Abschluss und Bericht']),
    note,
    el('div', { class: 'row spread' }, [button, exports]),
    hint,
  ]);
}

void boot();
