import './styles.css';

import { ApiError, api, connectEvents, newRequestId } from './api';
import { renderGraph, type GraphSelection } from './graph';
import { createCard, el, legend, renderQuestion, renderTable, updateCard } from './render';
import { renderSettings } from './settings';
import type {
  AppConfig,
  Bezug,
  Entwurf,
  PingPongRun,
  HealthInfo,
  Job,
  JobContext,
  Marker,
  Relation,
  Session,
  SessionBundle,
  SparkEntry,
  SparkKind,
  Summary,
} from './types';

const STORAGE_KEY = 'xpertentisch.session';
const THEME_KEY = 'xpertentisch.theme';

type Thema = 'system' | 'light' | 'dark';

function geltendesThema(): Thema {
  try {
    const wert = localStorage.getItem(THEME_KEY);
    if (wert === 'light' || wert === 'dark') return wert;
  } catch {
    /* ohne Speicher folgt die Systemeinstellung */
  }
  return 'system';
}

function setzeThema(mode: Thema): void {
  try {
    localStorage.setItem(THEME_KEY, mode);
  } catch {
    /* ohne Speicher gilt die Wahl nur für diese Ansicht */
  }
  if (mode === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
}

function mitUebergang(arbeit: () => void): void {
  const reduziert = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const start = (
    document as Document & { startViewTransition?: (cb: () => void) => unknown }
  ).startViewTransition;
  if (reduziert || typeof start !== 'function') {
    arbeit();
    return;
  }
  start.call(document, arbeit);
}

interface AppState {
  config: AppConfig | null;
  health: HealthInfo | null;
  bundle: SessionBundle | null;
  selectedModels: Set<string>;
  streamOpen: boolean;
  settingsOpen: boolean;
  /** Worauf sich die nächste Eingabe bezieht. */
  bezug: Bezug | null;
  /** Nicht gesendete Gedanken — gehen bei Netzfehlern nicht verloren. */
  entwuerfe: Entwurf[];
  /** Laufende und beendete Wechselgespräche. */
  pingpong: PingPongRun[];
  /** Kuratierung für die nächste Eingabe. */
  curate: boolean;
  /** Alle Sitzungen, für die Auswahl im Kopf. */
  sessions: Session[];
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
  bezug: null,
  entwuerfe: [],
  pingpong: [],
  curate: false,
  sessions: [],
  pendingRequestId: null,
};

const ENTWURF_KEY = 'xpertentisch.entwuerfe';
const EINGABE_KEY = 'xpertentisch.eingabe';

function ladeEntwuerfe(): Entwurf[] {
  try {
    return JSON.parse(localStorage.getItem(ENTWURF_KEY) ?? '[]') as Entwurf[];
  } catch {
    return [];
  }
}

function speichereEntwuerfe(): void {
  try {
    localStorage.setItem(ENTWURF_KEY, JSON.stringify(state.entwuerfe));
  } catch {
    /* Ohne Speicher lebt der Entwurf nur in dieser Ansicht. */
  }
}

let disconnect: (() => void) | null = null;
const blocks = new Map<string, HTMLElement>();
/** Das Einstellungsfeld überlebt Neuzeichnungen — sonst wären Eingaben weg. */
let settingsPanel: HTMLElement | null = null;

const root = document.getElementById('app')!;

// --------------------------------------------------------------------- Start

async function boot(): Promise<void> {
  setzeThema(geltendesThema());
  try {
    const [config, health] = await Promise.all([api.config(), api.health()]);
    state.config = config;
    state.health = health;
    state.selectedModels = new Set(config.models.map((m) => m.id));
  } catch (error) {
    root.replaceChildren(
      el('div', { class: 'flaeche' }, [
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
    state.bundle = {
      session, sparks: [], relations: [], pending: false,
      last_event_id: 0, exported_at: Date.now() / 1000,
    };
  }
  state.entwuerfe = ladeEntwuerfe();
  rememberSession(state.bundle.session.id);
  void ladeSitzungen();
  void ladePingPong();
  renderShell();
  openStream();
  void sendeEntwuerfe();
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
  verwerfeNeueBeitraege();

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
      el('div', { class: 'flaeche banner' }, [
        'Achtung: Test-Provider sind aktiv. Die Antworten stammen nicht von echten Modellen.',
      ]),
    );
  }
  root.append(providerWarnings());
  root.append(entwurfsliste());
  root.append(sparkForm());
  root.append(pingpongPanel());
  zeichnePingPong();

  const stream = el('div', { id: 'sparks' });
  root.append(stream);
  blocks.clear();
  for (const entry of bundle.sparks) {
    const block = renderSparkBlock(entry);
    blocks.set(entry.spark.id, block);
    stream.append(block);
  }
  if (bundle.sparks.length === 0) {
    stream.append(
      el('div', { class: 'tisch-leer', id: 'tisch-leer' }, [
        el('p', { class: 'leer-zeile' }, ['Noch keine Stimme am Tisch.']),
        el('p', { class: 'hint' }, [
          'Ein Funke oben. Die Modelle antworten unabhängig — nichts wird verrechnet, nichts gewinnt.',
        ]),
      ]),
    );
  }

  root.append(closingSection());
  root.append(
    el('footer', { class: 'foot' }, [
      'XPERTENTiSCH — mehrere Modelle antworten unabhängig auf denselben Funken. ' +
        'Keine Rangfolge, keine verbindliche Lehre; die Einordnung bleibt bei dir.',
    ]),
  );
}

async function ladeSitzungen(): Promise<void> {
  try {
    state.sessions = (await api.listSessions()).sessions;
    zeichneSitzungswahl();
  } catch {
    /* Die Liste ist Beiwerk; ohne sie arbeitet der Tisch weiter. */
  }
}

async function ladePingPong(): Promise<void> {
  try {
    state.pingpong = (await api.listPingPong(state.bundle!.session.id)).runs;
    zeichnePingPong();
  } catch {
    /* ohne Liste geht es auch */
  }
}

/** Zeichnet nur die Liste der Läufe neu — das Formular darüber bleibt, wie es ist. */
function zeichnePingPong(): void {
  const liste = document.getElementById('pp-laeufe');
  if (!liste) return;
  liste.replaceChildren();
  for (const lauf of state.pingpong) {
    const zeile = el('div', { class: `pingpong-lauf ${lauf.status}` }, [
      el('p', { class: 'hint' }, [
        `${lauf.labels.join(' ↔ ')} · Beitrag ${lauf.turn} von ${lauf.max_turns} · ` +
          `${PINGPONG_STAND[lauf.status] ?? lauf.status}` +
          `${lauf.stopped_reason ? ` (${lauf.stopped_reason})` : ''}`,
      ]),
    ]);
    if (lauf.status === 'laeuft') {
      const stopp = el('button', { type: 'button', class: 'gefahr' }, ['Stoppen']);
      stopp.addEventListener('click', async () => {
        stopp.disabled = true;
        try {
          await api.stopPingPong(state.bundle!.session.id, lauf.id);
          await ladePingPong();
          zeichnePingPong();
        } catch {
          stopp.disabled = false;
        }
      });
      zeile.append(stopp);
    }
    liste.append(zeile);
  }
}

const PINGPONG_STAND: Record<string, string> = {
  laeuft: 'läuft',
  gestoppt: 'gestoppt',
  beendet: 'beendet',
};

/** Schickt liegengebliebene Gedanken nach — mit derselben Kennung wie vorher. */
async function sendeEntwuerfe(): Promise<void> {
  if (state.entwuerfe.length === 0) return;
  const bundle = state.bundle!;
  for (const entwurf of [...state.entwuerfe]) {
    try {
      const ergebnis = await api.createSpark(
        bundle.session.id, entwurf.prompt, entwurf.clientRequestId,
        entwurf.modelIds, entwurf.refs, entwurf.kind, entwurf.curate,
      );
      state.entwuerfe = state.entwuerfe.filter(
        (e) => e.clientRequestId !== entwurf.clientRequestId,
      );
      speichereEntwuerfe();
      if (!ergebnis.duplicate) ensureSpark({ spark: ergebnis.spark, jobs: ergebnis.jobs, markers: [], summary: null });
    } catch (fehler) {
      entwurf.lastError = (fehler as Error).message;
      speichereEntwuerfe();
      break; // Erst wieder versuchen, wenn die Verbindung steht.
    }
  }
  renderShell();
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
    mitUebergang(() => {
      state.settingsOpen = !state.settingsOpen;
      renderShell();
    });
    if (state.settingsOpen) {
      document.getElementById('einstellungen')?.scrollIntoView({ block: 'nearest' });
    }
  });

  return el('header', { class: 'top' }, [
    el('div', { class: 'headline' }, [
      title,
      el('p', { class: 'sub' }, ['Mobiler TiSCH']),
      sitzungswahl(),
    ]),
    el('div', { class: 'row' }, [status, themaWahl(), zahnrad]),
  ]);
}

function themaWahl(): HTMLElement {
  const wrap = el('div', { class: 'thema-wahl', role: 'group', 'aria-label': 'Darstellung' });
  const aktuell = geltendesThema();
  const knoepfe: Array<[Thema, string]> = [
    ['system', 'System'],
    ['light', 'Hell'],
    ['dark', 'Dunkel'],
  ];
  for (const [mode, label] of knoepfe) {
    const knopf = el('button', {
      type: 'button',
      'aria-pressed': aktuell === mode ? 'true' : 'false',
    }, [label]);
    knopf.addEventListener('click', () => {
      setzeThema(mode);
      wrap.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', 'false'));
      knopf.setAttribute('aria-pressed', 'true');
    });
    wrap.append(knopf);
  }
  return wrap;
}

/** Zwischen Sitzungen wechseln oder eine neue beginnen. */
function sitzungswahl(): HTMLElement {
  const zeile = el('div', { class: 'row sitzungen' });
  const auswahl = el('select', {
    id: 'sitzungswahl', 'aria-label': 'Sitzung wählen',
  }) as HTMLSelectElement;
  zeile.append(auswahl, (() => {
    const neu = el('button', { type: 'button' }, ['Neue Sitzung']);
    neu.addEventListener('click', async () => {
      neu.disabled = true;
      try {
        const sitzung = await api.createSession(defaultTitle());
        await wechsleSitzung(sitzung.id);
      } finally {
        neu.disabled = false;
      }
    });
    return neu;
  })());
  auswahl.addEventListener('change', () => void wechsleSitzung(auswahl.value));
  zeichneSitzungswahl(auswahl);
  return zeile;
}

function zeichneSitzungswahl(node?: HTMLSelectElement | null): void {
  const auswahl = node ?? document.getElementById('sitzungswahl') as HTMLSelectElement | null;
  if (!auswahl || !state.bundle) return;
  const aktuell = state.bundle.session;
  const alle = state.sessions.some((s) => s.id === aktuell.id)
    ? state.sessions
    : [aktuell, ...state.sessions];
  auswahl.replaceChildren();
  for (const sitzung of alle) {
    const option = document.createElement('option');
    option.value = sitzung.id;
    option.textContent =
      sitzung.title + (sitzung.status === 'abgeschlossen' ? ' · abgeschlossen' : '');
    option.selected = sitzung.id === aktuell.id;
    auswahl.append(option);
  }
}

async function wechsleSitzung(sessionId: string): Promise<void> {
  if (!sessionId || sessionId === state.bundle?.session.id) return;
  try {
    state.bundle = await api.getSession(sessionId);
    state.bezug = null;
    state.pingpong = [];
    rememberSession(sessionId);
    mitUebergang(() => renderShell());
    openStream();
    void ladePingPong();
  } catch (fehler) {
    window.alert(`Sitzung konnte nicht geöffnet werden: ${(fehler as Error).message}`);
  }
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
  const banner = el('div', { class: 'flaeche banner' }, [
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

// --------------------------------------------------------- Offene Entwürfe

/** Was noch nicht beim Server ankam — sichtbar, nicht verloren. */
function entwurfsliste(): HTMLElement {
  const behaelter = el('div', {});
  if (state.entwuerfe.length === 0) return behaelter;

  const panel = el('div', { class: 'flaeche banner', id: 'entwuerfe' }, [
    el('p', { class: 'hint' }, [
      `${state.entwuerfe.length} Gedanke(n) nur auf diesem Gerät vorgemerkt — ` +
        'noch nicht beim Dienst angekommen. Nichts davon geht verloren.',
    ]),
  ]);
  for (const entwurf of state.entwuerfe) {
    panel.append(
      el('div', { class: 'entwurf' }, [
        el('p', { class: 'question' }, [entwurf.prompt]),
        entwurf.lastError ? el('p', { class: 'hint error' }, [entwurf.lastError]) : el('span', {}),
      ]),
    );
  }
  const zeile = el('div', { class: 'row' });
  const nochmal = el('button', { class: 'primary', type: 'button', id: 'entwuerfe-senden' },
    ['Jetzt senden']);
  nochmal.addEventListener('click', () => void sendeEntwuerfe());
  const verwerfen = el('button', { type: 'button', class: 'gefahr' }, ['Verwerfen']);
  verwerfen.addEventListener('click', () => {
    if (!window.confirm('Alle vorgemerkten Gedanken verwerfen?')) return;
    state.entwuerfe = [];
    speichereEntwuerfe();
    renderShell();
  });
  zeile.append(nochmal, verwerfen);
  panel.append(zeile);
  behaelter.append(panel);
  return behaelter;
}

// ------------------------------------------------------------- Ping-Pong

/** Ein begrenztes Wechselgespräch — die Grenzen stehen vorher sichtbar da. */
function pingpongPanel(): HTMLElement {
  const behaelter = el('div', {});
  const modelle = state.config?.models ?? [];
  const bundle = state.bundle!;
  if (modelle.length < 2 || bundle.session.status !== 'offen') return behaelter;

  const laufend = state.pingpong.filter((r) => r.status === 'laeuft');
  const panel = el('details', { class: 'flaeche pingpong', id: 'pingpong' });
  panel.open = laufend.length > 0;
  panel.append(el('summary', {}, ['Wechselgespräch zwischen Modellen']));
  panel.append(el('div', { id: 'pp-laeufe' }));

  const auftrag = el('textarea', {
    id: 'pp-prompt', rows: '2',
    placeholder: 'Untersuchungsauftrag für das Wechselgespräch …',
    'aria-label': 'Untersuchungsauftrag',
  }) as HTMLTextAreaElement;

  const teilnehmer = el('div', { class: 'modelpicker' });
  const gewaehlt = new Set(modelle.slice(0, 2).map((m) => m.id));
  for (const modell of modelle) {
    const box = el('input', { type: 'checkbox', value: modell.id }) as HTMLInputElement;
    box.checked = gewaehlt.has(modell.id);
    box.addEventListener('change', () => {
      if (box.checked) gewaehlt.add(modell.id);
      else gewaehlt.delete(modell.id);
      zeigeGrenzen();
    });
    teilnehmer.append(el('label', {}, [box, modell.label]));
  }

  const runden = el('input', {
    type: 'number', min: '2', max: '12', id: 'pp-runden',
    'aria-label': 'Höchstzahl zusätzlicher Beiträge',
  }) as HTMLInputElement;
  runden.value = '4';
  runden.addEventListener('input', () => zeigeGrenzen());

  const grenzen = el('p', { class: 'hint', id: 'pp-grenzen' }, []);
  function zeigeGrenzen(): void {
    const namen = modelle.filter((m) => gewaehlt.has(m.id)).map((m) => m.label);
    grenzen.textContent =
      `Vor dem Start: ${namen.join(' ↔ ') || '— niemand gewählt —'}; ` +
      `höchstens ${runden.value} zusätzliche Beiträge; ` +
      `Zeitgrenze je Beitrag wie in den Einstellungen. ` +
      (state.bezug ? `Bezug: ${state.bezug.label}.` : 'Ohne Bezugsbeitrag.');
  }
  zeigeGrenzen();

  const starten = el('button', { class: 'primary', type: 'button', id: 'pp-start' },
    ['Wechselgespräch starten']);
  const meldung = el('p', { class: 'hint', id: 'pp-meldung' }, []);
  starten.addEventListener('click', async () => {
    if (!auftrag.value.trim()) {
      meldung.className = 'hint error';
      meldung.textContent = 'Ohne Untersuchungsauftrag geht es nicht.';
      return;
    }
    starten.disabled = true;
    try {
      await api.startPingPong(bundle.session.id, {
        prompt: auftrag.value.trim(),
        refs: state.bezug ? [state.bezug.id] : [],
        participants: [...gewaehlt],
        max_turns: Number(runden.value),
      });
      auftrag.value = '';
      await ladePingPong();
    } catch (fehler) {
      meldung.className = 'hint error';
      meldung.textContent = (fehler as Error).message;
    } finally {
      starten.disabled = false;
    }
  });

  panel.append(
    el('p', { class: 'hint' }, [
      'Die Modelle antworten abwechselnd aufeinander. Es läuft nur so weit, wie du ' +
        'es hier festlegst — und lässt sich jederzeit stoppen.',
    ]),
    auftrag,
    el('div', { class: 'feld' }, [
      el('label', {}, ['Beteiligte Stimmen']), teilnehmer,
    ]),
    el('div', { class: 'feld' }, [
      el('label', { for: 'pp-runden' }, ['Höchstzahl zusätzlicher Beiträge']), runden,
    ]),
    grenzen,
    el('div', { class: 'row' }, [starten]),
    meldung,
  );
  behaelter.append(panel);
  return behaelter;
}

// ------------------------------------------------------------------- Bezüge

const BEZUG_KNOPF: Record<SparkKind, string> = {
  funke: 'Funke setzen',
  antwort: 'Antwort senden',
  weitergabe: 'Zur Prüfung geben',
  gegenposition: 'Gegenposition anfragen',
  vertiefung: 'Strang vertiefen',
  pingpong: 'Wechselgespräch',
  kuratierung: 'Kuratierung',
};

const FUNKE_ART: Record<SparkKind, string> = {
  funke: '',
  antwort: 'Antwort',
  weitergabe: 'Weitergabe',
  gegenposition: 'Gegenposition',
  vertiefung: 'Vertiefung',
  pingpong: 'Wechselgespräch',
  kuratierung: 'Kuratierung (maschinell)',
};

/** Zeigt, worauf sich die nächste Eingabe bezieht — und lässt es lösen. */
function bezugsleiste(): HTMLElement {
  const leiste = el('div', { class: 'bezugsleiste', id: 'bezugsleiste' });
  if (!state.bezug) {
    leiste.hidden = true;
    return leiste;
  }
  const loesen = el('button', { type: 'button', class: 'iconbutton' }, ['✕']);
  loesen.setAttribute('aria-label', 'Bezug aufheben');
  loesen.addEventListener('click', () => {
    state.bezug = null;
    renderShell();
  });
  leiste.append(
    el('span', { class: 'tag' }, ['bezieht sich auf']),
    el('span', { class: 'bezug-name' }, [state.bezug.label]),
    el('span', { class: 'hint' }, [state.bezug.hint]),
    loesen,
  );
  return leiste;
}

/** Setzt den Bezug und bringt die Eingabe in den Blick. */
function setzeBezug(bezug: Bezug, vorschlag = ''): void {
  state.bezug = bezug;
  renderShell();
  const feld = document.getElementById('prompt') as HTMLTextAreaElement | null;
  if (feld) {
    if (vorschlag && !feld.value.trim()) feld.value = vorschlag;
    feld.focus();
    feld.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

/** Die Handlungen an einer Modellkarte: antworten, weitergeben, vertiefen. */
function kartenAktionen(job: Job, entry: SparkEntry): HTMLElement {
  const zeile = el('div', { class: 'row card-actions' });
  const andere = (state.config?.models ?? []).filter((m) => m.id !== job.model_id);

  const knopf = (text: string, bauen: () => void) => {
    const b = el('button', { type: 'button' }, [text]);
    b.addEventListener('click', bauen);
    zeile.append(b);
  };

  knopf('Antworten', () =>
    setzeBezug({
      id: job.id, label: `${job.label}, Funke ${entry.spark.seq}`, kind: 'antwort',
      hint: 'geht an alle Modelle am Tisch',
    }),
  );

  for (const ziel of andere) {
    knopf(`An ${ziel.label} geben`, () =>
      setzeBezug(
        {
          id: job.id, label: `${job.label} → ${ziel.label}`, kind: 'weitergabe',
          modelId: ziel.id, hint: `nur ${ziel.label} antwortet`,
        },
        'Prüfe diese Aussage kritisch.',
      ),
    );
  }

  knopf('Gegenposition', () =>
    setzeBezug(
      {
        id: job.id, label: `Gegenposition zu ${job.label}`, kind: 'gegenposition',
        hint: 'geht an alle Modelle am Tisch',
      },
      'Welche begründete Gegenposition gibt es dazu?',
    ),
  );

  knopf('Strang vertiefen', () =>
    setzeBezug({
      id: job.id, label: `Vertiefung von ${job.label}`, kind: 'vertiefung',
      hint: 'geht an alle Modelle am Tisch',
    }),
  );

  return zeile;
}

/** „Worauf antwortet diese Stimme?“ — erst beim Aufklappen geladen. */
function kontextAnsicht(job: Job): HTMLElement {
  const block = el('details', { class: 'kontext' });
  block.append(el('summary', {}, ['Worauf antwortet diese Stimme?']));
  const inhalt = el('div', { class: 'kontext-inhalt' }, [
    el('p', { class: 'hint' }, ['wird geladen …']),
  ]);
  block.append(inhalt);

  let geladen = false;
  block.addEventListener('toggle', async () => {
    if (!block.open || geladen) return;
    geladen = true;
    try {
      zeichneKontext(inhalt, await api.jobContext(job.id));
    } catch (fehler) {
      geladen = false;
      inhalt.replaceChildren(
        el('p', { class: 'hint' }, [(fehler as Error).message]),
      );
    }
  });
  return block;
}

function zeichneKontext(ziel: HTMLElement, kontext: JobContext): void {
  ziel.replaceChildren();
  if (kontext.entries.length === 0) {
    ziel.append(
      el('p', { class: 'hint' }, [
        'Nur der Funke selbst — dieser Auftrag hatte keinen weiteren Gesprächsauszug.',
      ]),
    );
  } else {
    const liste = el('ul', { class: 'kontext-liste' });
    for (const eintrag of kontext.entries) {
      const gewaehlt = eintrag.reason === 'ausdrücklich gewählt';
      liste.append(
        el('li', { class: gewaehlt ? 'gewaehlt' : '' }, [
          el('span', { class: 'tag' }, [gewaehlt ? 'gewählt' : 'Verlauf']),
          ` ${eintrag.label}`,
          el('span', { class: 'hint' }, [` · ${eintrag.chars} Zeichen`]),
        ]),
      );
    }
    ziel.append(liste);
  }
  if (kontext.truncated) {
    ziel.append(
      el('p', { class: 'hint error' }, [
        'Der Auszug wurde gekürzt; ausdrücklich gewählte Bezüge blieben vollständig.',
      ]),
    );
  }
  const roh = el('details', { class: 'kontext-roh' });
  roh.append(el('summary', {}, ['Übergebener Wortlaut']));
  roh.append(el('pre', { class: 'kontext-text' }, [kontext.rendered]));
  ziel.append(roh, el('p', { class: 'hint' }, [`Regel: ${kontext.rule}`]));
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

  const send = el('button', { class: 'primary', type: 'submit' }, [
    state.bezug ? BEZUG_KNOPF[state.bezug.kind] : 'Funke setzen',
  ]);
  const message = el('p', { class: 'hint', id: 'spark-message' }, []);

  const zeile = el('div', { class: 'row spread' }, [picker]);
  const kurator = state.config?.curator;
  if (kurator) {
    const schalter = el('input', { type: 'checkbox', id: 'kuratierung' }) as HTMLInputElement;
    schalter.checked = state.curate;
    schalter.addEventListener('change', () => {
      state.curate = schalter.checked;
    });
    picker.append(
      el('label', { class: 'schalter', for: 'kuratierung',
                    title: 'Ein zusätzlicher Modellaufruf vor der Runde' },
        [schalter, `Kuratierung durch ${kurator.label}`]),
    );
  }
  zeile.append(send);

  const form = el('form', { class: 'flaeche spark' }, [
    bezugsleiste(),
    textarea,
    zeile,
    message,
  ]);

  if (config.models.length === 0) {
    // Erster Start: noch kein Anbieter eingerichtet. Den Weg zeigen, statt
    // ein Formular anzubieten, das nur in eine Fehlermeldung laufen kann.
    (textarea as HTMLTextAreaElement).disabled = true;
    (send as HTMLButtonElement).disabled = true;
    message.textContent =
      'Noch sitzt niemand am Tisch. In den Einstellungen mindestens einen Anbieter ' +
      'einschalten und mit einem Schlüssel versehen.';
    const hin = el('button', { type: 'button', class: 'primary' }, ['Anbieter einrichten']);
    hin.addEventListener('click', () => {
      state.settingsOpen = true;
      renderShell();
      document.getElementById('einstellungen')?.scrollIntoView({ block: 'nearest' });
    });
    picker.append(hin);
    return form;
  }

  if (closed) {
    (textarea as HTMLTextAreaElement).disabled = true;
    (send as HTMLButtonElement).disabled = true;
    message.textContent = 'Die Sitzung ist abgeschlossen. Neue Funken sind nicht mehr möglich.';
  }

  // Getipptes überlebt ein Neuladen.
  try {
    const gemerkt = localStorage.getItem(EINGABE_KEY);
    if (gemerkt && !(textarea as HTMLTextAreaElement).value) {
      (textarea as HTMLTextAreaElement).value = gemerkt;
    }
  } catch {
    /* ohne Speicher eben nicht */
  }
  textarea.addEventListener('input', () => {
    try {
      localStorage.setItem(EINGABE_KEY, (textarea as HTMLTextAreaElement).value);
    } catch {
      /* ohne Speicher eben nicht */
    }
  });

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
    const bezug = state.bezug;
    const entwurf: Entwurf = {
      clientRequestId: state.pendingRequestId,
      prompt,
      refs: bezug ? [bezug.id] : [],
      kind: bezug?.kind ?? 'funke',
      modelIds: bezug?.modelId ? [bezug.modelId] : [...state.selectedModels],
      curate: state.curate,
      createdAt: Date.now(),
      lastError: '',
    };
    // Erst vormerken, dann senden: ein Netzfehler vernichtet keinen Gedanken.
    state.entwuerfe = [...state.entwuerfe, entwurf];
    speichereEntwuerfe();
    (textarea as HTMLTextAreaElement).value = '';
    try {
      localStorage.removeItem(EINGABE_KEY);
    } catch {
      /* ohne Speicher eben nicht */
    }

    try {
      const result = await api.createSpark(
        bundle.session.id, entwurf.prompt, entwurf.clientRequestId,
        entwurf.modelIds, entwurf.refs, entwurf.kind, entwurf.curate,
      );
      state.pendingRequestId = null;
      state.bezug = null;
      state.entwuerfe = state.entwuerfe.filter(
        (e) => e.clientRequestId !== entwurf.clientRequestId,
      );
      speichereEntwuerfe();
      const hinweise: string[] = [];
      if (result.duplicate) {
        hinweise.push('Dieser Funke lief bereits — es wurden keine neuen Aufträge gestartet.');
      }
      if (result.curation && !result.curation.gestartet) {
        hinweise.push(String(result.curation.grund ?? ''));
      }
      message.textContent = hinweise.join(' ');
      ensureSpark({ spark: result.spark, jobs: result.jobs, markers: [], summary: null });
      renderShell();
    } catch (error) {
      state.pendingRequestId = null;
      const detail = error instanceof ApiError ? error.message : (error as Error).message;
      entwurf.lastError = detail;
      speichereEntwuerfe();
      message.className = 'hint error';
      message.textContent =
        `Nicht gesendet: ${detail} Der Gedanke ist vorgemerkt und geht nicht verloren.`;
      renderShell();
    } finally {
      (send as HTMLButtonElement).disabled = bundle.session.status !== 'offen';
    }
  });

  return form;
}

// -------------------------------------------------------------------- Funken

function renderSparkBlock(entry: SparkEntry): HTMLElement {
  const kopf = el('h2', {}, [`Funke ${entry.spark.seq}`]);
  const art = FUNKE_ART[entry.spark.kind] ?? '';
  if (art) kopf.append(el('span', { class: 'tag' }, [art]));
  const block = el('section', {
    class: `spark-block art-${entry.spark.kind}`, 'data-spark-id': entry.spark.id,
  }, [kopf, renderQuestion(entry)]);

  const cards = el('div', { class: 'cards' });
  for (const job of entry.jobs) {
    const card = createCard(job, entry.markers);
    ruesteKarteAus(card, job, entry);
    cards.append(card);
  }
  block.append(cards);

  const panels = el('div', { class: 'panel-grid' });
  block.append(panels);
  if (entry.summary) renderPanels(block, entry.summary);
  return block;
}

/** Hängt Aktionen und Kontextansicht an eine Karte. Beides überlebt Aktualisierungen. */
function ruesteKarteAus(card: HTMLElement, job: Job, entry: SparkEntry): void {
  const fuss = card.querySelector('.card-foot');
  if (!fuss) return;
  fuss.replaceChildren();
  if (job.status === 'not_requested') return;
  if (job.status === 'queued' || job.status === 'running' || job.status === 'streaming') {
    fuss.append(abbruchKnopf(job));
  }
  if ((job.text ?? '').trim()) fuss.append(kartenAktionen(job, entry));
  fuss.append(kontextAnsicht(job));
}

/** Bricht genau diesen Auftrag ab. Die anderen Modelle laufen weiter. */
function abbruchKnopf(job: Job): HTMLElement {
  const zeile = el('div', { class: 'row card-actions' });
  const knopf = el('button', { type: 'button', class: 'gefahr' }, ['Abbrechen']);
  const meldung = el('span', { class: 'hint' }, []);
  knopf.addEventListener('click', async () => {
    knopf.disabled = true;
    try {
      await api.cancelJob(job.id);
      // Der Zustandswechsel kommt über das Ereignis; hier ist nichts zu tun.
    } catch (fehler) {
      // Häufigster Fall: der Auftrag war schneller fertig als der Klick.
      knopf.disabled = false;
      meldung.className = 'hint';
      meldung.textContent = (fehler as Error).message;
    }
  });
  zeile.append(knopf, meldung);
  return zeile;
}

/** Maschinelle Bezüge — als Vorschlag, den man bestätigen oder verwerfen kann. */
function beziehungsPanel(entry: SparkEntry): HTMLElement | null {
  const jobIds = new Set(entry.jobs.map((j) => j.id));
  const eigene = (state.bundle?.relations ?? []).filter(
    (r) => r.origin === 'maschine' && jobIds.has(r.from_id) && jobIds.has(r.to_id),
  );
  if (eigene.length === 0) return null;

  const namen = new Map(entry.jobs.map((j) => [j.id, j.label]));
  const panel = el('section', { class: 'flaeche panel' }, [
    el('h4', {}, ['Gefundene Bezüge']),
    el('p', { class: 'hint' }, [
      'Vorschläge der Auswertung. Erst deine Bestätigung macht daraus einen Befund.',
    ]),
  ]);

  for (const bez of eigene) {
    const zeile = el('div', { class: `bezug-zeile ${bez.status}`, 'data-relation': bez.id });
    zeile.append(
      el('div', { class: 'row' }, [
        el('span', { class: `tag ${bez.type === 'widerspricht' ? 'error' : 'done'}` }, [
          bez.type === 'widerspricht' ? 'Widerspruch' : 'Übereinstimmung',
        ]),
        el('span', {}, [`${namen.get(bez.from_id) ?? bez.from_id} ↔ ${namen.get(bez.to_id) ?? bez.to_id}`]),
        el('span', { class: 'tag' }, [STATUS_BEZUG[bez.status]]),
      ]),
    );
    if (bez.note) zeile.append(el('p', { class: 'hint' }, [bez.note]));

    const knoepfe = el('div', { class: 'row' });
    for (const [text, status] of [
      ['Bestätigen', 'bestaetigt'],
      ['Verwerfen', 'abgelehnt'],
    ] as const) {
      const b = el('button', { type: 'button' }, [text]);
      b.disabled = bez.status === status;
      b.addEventListener('click', async () => {
        try {
          const antwort = await api.setRelation(state.bundle!.session.id, bez.id, status);
          state.bundle!.relations = antwort.relations;
          renderShell();
        } catch (fehler) {
          b.after(el('span', { class: 'hint error' }, [(fehler as Error).message]));
        }
      });
      knoepfe.append(b);
    }
    zeile.append(knoepfe);
    panel.append(zeile);
  }
  return panel;
}

const STATUS_BEZUG: Record<Relation['status'], string> = {
  vorschlag: 'Vorschlag',
  bestaetigt: 'von dir bestätigt',
  abgelehnt: 'von dir verworfen',
};

function renderPanels(block: HTMLElement, summary: Summary): void {
  const panels = block.querySelector('.panel-grid');
  if (!panels) return;
  panels.replaceChildren();
  panels.append(renderTable(summary));

  const graphPanel = el('section', { class: 'flaeche panel' }, [el('h4', {}, ['Beziehungsnetz'])]);
  const wrap = el('div', { class: 'graphwrap' });
  wrap.append(renderGraph(summary, (selection) => openAnswers(block, selection)));
  graphPanel.append(wrap, legend());
  graphPanel.append(
    el('p', { class: 'hint' }, [
      'Knoten oder Kante anklicken, um die zugehörigen Antworten zu öffnen.',
    ]),
  );
  panels.append(graphPanel);

  const spark = state.bundle?.sparks.find((e) => e.spark.id === summary.spark_id);
  if (spark) {
    const bezuege = beziehungsPanel(spark);
    if (bezuege) panels.append(bezuege);
  }
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
  meldeNeuenBeitrag(block);
  return block;
}

/** Zählt neu eingetroffene Beiträge, die unterhalb des Sichtfelds liegen.
 *
 * Es wird niemals ungefragt gescrollt: wer gerade liest, bleibt, wo er ist.
 * Der Hinweis bringt einen erst auf Klick nach unten.
 */
let neueBeitraege: HTMLElement | null = null;

function meldeNeuenBeitrag(block: HTMLElement): void {
  const kasten = block.getBoundingClientRect();
  if (kasten.top < window.innerHeight) return; // schon sichtbar

  const hinweis = neueBeitraege ?? el('button', {
    type: 'button', class: 'neue-beitraege', id: 'neue-beitraege',
  }, []);
  const bisher = Number(hinweis.dataset.anzahl ?? '0') + 1;
  hinweis.dataset.anzahl = String(bisher);
  hinweis.textContent = bisher === 1 ? '1 neuer Beitrag ↓' : `${bisher} neue Beiträge ↓`;
  if (!neueBeitraege) {
    hinweis.addEventListener('click', () => {
      const ziel = document.getElementById('sparks')?.lastElementChild;
      ziel?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      verwerfeNeueBeitraege();
    });
    document.body.append(hinweis);
    neueBeitraege = hinweis;
  }
}

function verwerfeNeueBeitraege(): void {
  neueBeitraege?.remove();
  neueBeitraege = null;
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
    case 'auftrag.teilstueck':
      // Der Zwischenstand ersetzt den bisherigen Text vollständig — der Server
      // schickt immer den ganzen bisher angefallenen Text, nicht nur das Neue.
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'streaming',
        text: String(payload.text ?? ''),
      }));
      break;
    case 'auftrag.abgebrochen':
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'cancelled',
        text: String(payload.text ?? job.text ?? ''),
        partial: Boolean(payload.partial),
        error: 'Von dir abgebrochen.',
      }));
      break;
    case 'auftrag.fertig':
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'done',
        text: String(payload.text ?? ''),
        partial: Boolean(payload.partial),
        latency_ms: (payload.latency_ms as number) ?? null,
        error: null,
        // Verbrauch und Kosten kommen mit dem Ereignis; ohne sie stünde die
        // Karte bis zum nächsten Neuladen ohne diese Angaben da.
        tokens_in: (payload.tokens_in as number) ?? null,
        tokens_out: (payload.tokens_out as number) ?? null,
        cost_micro: (payload.cost_micro as number) ?? null,
        cost_source: payload.cost_source === 'berechnet' ? 'berechnet' : 'unbekannt',
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
      if (Array.isArray(payload.relations)) {
        bundle.relations = payload.relations as Relation[];
      }
      const block = blocks.get(sparkId);
      if (block) {
        for (const job of entry.jobs) {
          const card = block.querySelector<HTMLElement>(
            `.card[data-job-id="${cssEscape(job.id)}"]`,
          );
          if (card) {
            updateCard(card, job, entry.markers);
            ruesteKarteAus(card, job, entry);
          }
        }
        renderPanels(block, entry.summary);
      }
      break;
    }
    case 'pingpong.gestartet':
    case 'pingpong.runde':
    case 'pingpong.ende': {
      const lauf = payload as unknown as PingPongRun;
      const index = state.pingpong.findIndex((r) => r.id === lauf.id);
      if (index >= 0) state.pingpong[index] = lauf;
      else state.pingpong.push(lauf);
      zeichnePingPong();
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
  if (card) {
    updateCard(card, entry.jobs[index], entry.markers);
    ruesteKarteAus(card, entry.jobs[index], entry);
  }
}

// ------------------------------------------------------------------ Abschluss

function closingSection(): HTMLElement {
  const bundle = state.bundle!;
  const closed = bundle.session.status === 'abgeschlossen';
  const base = `/api/sessions/${bundle.session.id}`;

  const exports = el('div', { class: 'row' }, [
    el('a', { class: 'btn', href: `${base}/report.html`, download: '' }, ['Bericht als HTML']),
    el('a', { class: 'btn', href: `${base}/report.md`, download: '' }, ['Bericht als Markdown']),
    el('a', { class: 'btn', href: `${base}/report.json`, download: '' }, ['Sitzung als JSON']),
  ]);

  if (closed) {
    const section = el('section', { class: 'flaeche closing closed-banner' }, [
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

  return el('section', { class: 'flaeche closing' }, [
    el('h4', {}, ['Abschluss und Bericht']),
    note,
    el('div', { class: 'row spread' }, [button, exports]),
    hint,
  ]);
}

void boot();
