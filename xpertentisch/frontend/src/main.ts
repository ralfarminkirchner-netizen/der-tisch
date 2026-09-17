import './styles.css';

import { ApiError, api, connectEvents, newRequestId } from './api';
import { renderGraph, type GraphSelection } from './graph';
import {
  LINSEN,
  VERLAUF_ERKLAERUNG,
  renderHerkunft,
  renderSzenario,
  renderThemen,
  renderVerlauf,
  renderZeit,
} from './linsen';
import type { LinsenArt } from './linsen';
import { createCard, el, legend, renderQuestion, renderTable, updateCard } from './render';
import { renderSettings } from './settings';
import type {
  AppConfig,
  Bezug,
  Entwurf,
  Folge,
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
  Szenario,
} from './types';

const STORAGE_KEY = 'xpertentisch.session';

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
  /** Welche Linse gerade auf die Daten gelegt ist. */
  linse: LinsenArt;
  /** Auf welchen Funken die Linsen schauen. Null heißt: den letzten. */
  gewaehlterSpark: string | null;
  /**
   * Fundstellen der aktuellen Auswahl, je Auftrag.
   *
   * Nicht gespeichert und nicht vom Server: sie entstehen beim Klick auf eine
   * genannte Folge und verschwinden mit der nächsten Auswahl wieder.
   */
  zusatzMarker: Map<string, Marker[]>;
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
  linse: 'stimmen',
  gewaehlterSpark: null,
  zusatzMarker: new Map(),
};

const ENTWURF_KEY = 'xpertentisch.entwuerfe';
const EINGABE_KEY = 'xpertentisch.eingabe';
const THEMA_KEY = 'xpertentisch.thema';
const LINSEN_KEY = 'xpertentisch.linse';

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
  // Vor allem anderen: eine ausdrücklich gewählte Themenwahl greift sofort,
  // sonst blitzte beim Laden kurz das Thema des Geräts auf.
  themaAnwenden();
  state.linse = gemerkteLinse();
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
      session, sparks: [], szenarien: [], relations: [], pending: false,
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
  if (bundle.sparks.length === 0) stream.append(gedeckterTisch());

  // Ohne einen einzigen Beitrag gäbe es nichts zu betrachten: eine leere
  // Linsenfläche wäre nur Möblierung.
  if (bundle.sparks.length > 0) {
    root.append(linsenFlaeche());
    zeichneLinsen();
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
    state.settingsOpen = !state.settingsOpen;
    renderShell();
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
    el('div', { class: 'row' }, [status, themenschalter(), zahnrad]),
  ]);
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
    mitUebergang(renderShell);
    openStream();
    void ladePingPong();
  } catch (fehler) {
    window.alert(`Sitzung konnte nicht geöffnet werden: ${(fehler as Error).message}`);
  }
}

/** Blendet einen echten Zustandswechsel über, wo der Browser das kann.
 *
 * Bewusst nur beim Wechsel der Sitzung: dort wechselt der ganze Inhalt, und
 * die Überblendung macht verständlich, dass man woanders ist. Während
 * Antworten einlaufen, wird NICHT überblendet — das verschöbe die Leseposition
 * dessen, der gerade liest. Kann der Browser es nicht, passiert dasselbe ohne
 * Überblendung; es geht dabei nichts verloren.
 */
function mitUebergang(zeichnen: () => void): void {
  const starten = (document as Document & {
    startViewTransition?: (cb: () => void) => unknown;
  }).startViewTransition;
  const ruhig = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  if (typeof starten !== 'function' || ruhig) {
    zeichnen();
    return;
  }
  starten.call(document, zeichnen);
}

// ------------------------------------------------------------------- Thema

/** Systemvorgabe, ausdrücklich hell oder ausdrücklich dunkel.
 *
 * Das Gestaltungssystem kennt diese drei Zustände; ohne Schalter wären die
 * beiden ausdrücklichen Zustände nicht erreichbar. Eine ausdrückliche Wahl
 * hat Vorrang vor der Einstellung des Geräts und überdauert das Neuladen.
 */
type Thema = 'system' | 'light' | 'dark';

const THEMA_TEXT: Record<Thema, { zeichen: string; name: string }> = {
  system: { zeichen: '◐', name: 'Thema: dem Gerät folgen' },
  light: { zeichen: '☀', name: 'Thema: hell' },
  dark: { zeichen: '☾', name: 'Thema: dunkel' },
};

function gemerktesThema(): Thema {
  try {
    const wert = localStorage.getItem(THEMA_KEY);
    if (wert === 'light' || wert === 'dark') return wert;
  } catch {
    /* Ohne Speicher folgt das Thema dem Gerät. */
  }
  return 'system';
}

export function themaAnwenden(thema: Thema = gemerktesThema()): void {
  if (thema === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', thema);
}

function themenschalter(): HTMLElement {
  const folge: Thema[] = ['system', 'light', 'dark'];
  let aktuell = gemerktesThema();
  const knopf = el('button', {
    type: 'button', class: 'iconbutton', id: 'thema-schalter',
  }, []);

  const zeichnen = () => {
    knopf.textContent = THEMA_TEXT[aktuell].zeichen;
    knopf.setAttribute('aria-label', THEMA_TEXT[aktuell].name);
    knopf.setAttribute('title', `${THEMA_TEXT[aktuell].name} — zum Wechseln klicken`);
    knopf.dataset.thema = aktuell;
  };

  knopf.addEventListener('click', () => {
    aktuell = folge[(folge.indexOf(aktuell) + 1) % folge.length];
    try {
      if (aktuell === 'system') localStorage.removeItem(THEMA_KEY);
      else localStorage.setItem(THEMA_KEY, aktuell);
    } catch {
      /* Dann gilt die Wahl nur für diesen Besuch. */
    }
    themaAnwenden(aktuell);
    zeichnen();
  });

  zeichnen();
  return knopf;
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
  szenario: 'Folgen durchspielen',
  pingpong: 'Wechselgespräch',
  kuratierung: 'Kuratierung',
};

const FUNKE_ART: Record<SparkKind, string> = {
  funke: '',
  antwort: 'Antwort',
  weitergabe: 'Weitergabe',
  gegenposition: 'Gegenposition',
  vertiefung: 'Vertiefung',
  szenario: 'Szenario',
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
  const behaelter = el('div', { class: 'aktionen' });
  const zeile = el('div', { class: 'row card-actions' });
  const andere = (state.config?.models ?? []).filter((m) => m.id !== job.model_id);

  const knopf = (text: string, bauen: () => void, wohin: HTMLElement = zeile) => {
    const b = el('button', { type: 'button' }, [text]);
    b.addEventListener('click', bauen);
    wohin.append(b);
    return b;
  };

  // Die drei Handlungen, die sich an den ganzen Tisch richten, stehen offen da.
  knopf('Antworten', () =>
    setzeBezug({
      id: job.id, label: `${job.label}, Funke ${entry.spark.seq}`, kind: 'antwort',
      hint: 'geht an alle Modelle am Tisch',
    }),
  );

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

  // Folgen kommen von den Stimmen, nicht aus der Anwendung. Sie erscheinen als
  // eigener Beitrag und tauchen im Verlauf als eigener Zweig auf.
  knopf('Folgen durchspielen', () =>
    setzeBezug(
      {
        id: job.id, label: `Folgen von ${job.label}`, kind: 'szenario',
        hint: 'geht an alle Modelle am Tisch',
      },
      'Angenommen, das trifft zu: welche konkreten Folgen hätte es? Nenne die ' +
        'wahrscheinlichen und die unangenehmen, und sag, woran man früh merken ' +
        'würde, dass es anders kommt.',
    ),
  );
  behaelter.append(zeile);

  // Die Weitergabe an eine einzelne Stimme ist eine andere Art von Handlung —
  // und mit fünf Stimmen am Tisch wären es fünf weitere gleich aussehende
  // Knöpfe. Sie stehen darum zusammengefasst darunter, einen Griff entfernt.
  if (andere.length > 0) {
    const auswahl = el('details', { class: 'weitergabe' });
    auswahl.append(
      el('summary', {}, [
        andere.length === 1 ? 'An eine andere Stimme geben' : 'An eine Stimme geben',
      ]),
    );
    const ziele = el('div', { class: 'row' });
    for (const ziel of andere) {
      knopf(
        ziel.label,
        () =>
          setzeBezug(
            {
              id: job.id, label: `${job.label} → ${ziel.label}`, kind: 'weitergabe',
              modelId: ziel.id, hint: `nur ${ziel.label} antwortet`,
            },
            'Prüfe diese Aussage kritisch.',
          ),
        ziele,
      );
    }
    auswahl.append(ziele);
    behaelter.append(auswahl);
  }

  return behaelter;
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

/** Der leere Tisch: kein Formular ins Leere, sondern eine Ansage der Regeln.
 *
 * Es ist der erste Eindruck der Anwendung, und er soll sagen, worauf man sich
 * einlässt — nicht bloß, dass hier noch nichts steht.
 */
function gedeckterTisch(): HTMLElement {
  const anzahl = state.config?.models.length ?? 0;
  const namen = (state.config?.models ?? []).map((m) => m.label);
  return el('section', { class: 'flaeche leerer-tisch', id: 'leerer-tisch' }, [
    el('p', { class: 'gedeckt-zahl' }, [
      anzahl === 0 ? 'Niemand' : anzahl === 1 ? 'Eine Stimme' : `${anzahl} Stimmen`,
    ]),
    el('p', { class: 'gedeckt-satz' }, [
      anzahl === 0
        ? 'sitzt bisher am Tisch.'
        : 'am Tisch. Sie antworten unabhängig voneinander auf denselben Funken.',
    ]),
    anzahl > 0
      ? el('p', { class: 'hint' }, [namen.join(' · ')])
      : el('span', {}),
    el('p', { class: 'hint' }, [
      'Es entsteht keine Rangfolge und keine gemeinsame Antwort. Was ' +
        'übereinstimmt, was sich widerspricht und was nur einer sagt, wird ' +
        'markiert — einordnen musst du selbst.',
    ]),
  ]);
}

function renderSparkBlock(entry: SparkEntry): HTMLElement {
  const kopf = el('h2', {}, [`Funke ${entry.spark.seq}`]);
  const art = FUNKE_ART[entry.spark.kind] ?? '';
  if (art) kopf.append(el('span', { class: 'tag' }, [art]));
  const block = el('section', {
    class: `spark-block art-${entry.spark.kind}`, 'data-spark-id': entry.spark.id,
  }, [kopf, renderQuestion(entry)]);

  const cards = el('div', { class: 'cards' });
  for (const job of entry.jobs) {
    const card = createCard(job, markerFuer(entry, job.id));
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
  const laeuft =
    job.status === 'queued' || job.status === 'running' || job.status === 'streaming';
  if ((job.text ?? '').trim()) {
    const aktionen = kartenAktionen(job, entry);
    // Der Abbruch gehört in dieselbe Zeile, aber ans Ende: er ist die einzige
    // Handlung hier, die etwas beendet.
    if (laeuft) aktionen.querySelector('.card-actions')?.append(abbruchKnopf(job));
    fuss.append(aktionen);
  } else if (laeuft) {
    fuss.append(el('div', { class: 'row card-actions' }, [abbruchKnopf(job)]));
  }
  fuss.append(kontextAnsicht(job));
}

/** Bricht genau diesen Auftrag ab. Die anderen Modelle laufen weiter. */
function abbruchKnopf(job: Job): HTMLElement {
  const gruppe = el('span', { class: 'abbruch' });
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
  gruppe.append(knopf, meldung);
  return gruppe;
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

/**
 * Was unter einem Funken steht: der Vergleich und die gefundenen Bezüge.
 *
 * Die Linsen sind hier ausgezogen. Sie sitzen jetzt in einer eigenen Fläche,
 * die den ganzen Verlauf und den gewählten Funken zusammen zeigt — mit fünf
 * Linsen wäre das Panel unter jedem Funken zu eng geworden, und der Verlauf
 * war dort ohnehin nur zu Gast: er gilt für die ganze Sitzung.
 */
function renderPanels(block: HTMLElement, summary: Summary): void {
  const panels = block.querySelector('.panel-grid');
  if (!panels) return;
  panels.replaceChildren();
  panels.append(renderTable(summary));

  const spark = state.bundle?.sparks.find((e) => e.spark.id === summary.spark_id);
  if (spark) {
    const bezuege = beziehungsPanel(spark);
    if (bezuege) panels.append(bezuege);
  }
  zeichneLinsen();
}

/** Welche Linse zuletzt gewählt war. Überdauert das Neuladen. */
function gemerkteLinse(): LinsenArt {
  try {
    const wert = localStorage.getItem(LINSEN_KEY);
    if (LINSEN.some((l) => l.id === wert)) return wert as LinsenArt;
  } catch {
    /* ohne Speicher gilt die Vorgabe */
  }
  return 'stimmen';
}

/** Der Funke, auf den die Linsen gerade schauen — der letzte, solange keiner gewählt ist. */
function gewaehlterEintrag(): SparkEntry | null {
  const sparks = state.bundle?.sparks ?? [];
  if (sparks.length === 0) return null;
  const gewaehlt = sparks.find((e) => e.spark.id === state.gewaehlterSpark);
  return gewaehlt ?? sparks[sparks.length - 1];
}

/** Die Namen aller Beiträge — für die Herkunfts-Linse, die über Funken hinweg zeigt. */
function beitragsNamen(): Map<string, string> {
  const namen = new Map<string, string>();
  for (const eintrag of state.bundle?.sparks ?? []) {
    namen.set(eintrag.spark.id, `Funke ${eintrag.spark.seq}`);
    for (const job of eintrag.jobs) {
      namen.set(job.id, `${job.label} · F${eintrag.spark.seq}`);
    }
  }
  return namen;
}

/**
 * Die Linsenfläche: der ganze Verlauf und der gewählte Funke nebeneinander.
 *
 * Links steht der Verlauf der Sitzung. Er ist keine Linse unter mehreren,
 * sondern der Weg durch die Sitzung — und zugleich die Auswahl: ein Klick auf
 * einen Zweig entscheidet, worauf die gewählte Linse rechts schaut. So bleibt
 * sichtbar, wo man ist, während man die Sichtweise wechselt.
 */
function linsenFlaeche(): HTMLElement {
  const panel = el('section', { class: 'flaeche linsenraum', id: 'linsen' });
  panel.append(
    el('div', { class: 'row spread' }, [
      el('h3', {}, ['Linsen']),
      el('p', { class: 'hint' }, [
        'Dieselben Daten, verschieden gelesen. Keine Linse rechnet etwas hinzu.',
      ]),
    ]),
  );

  const raum = el('div', { class: 'linsenraster' });
  const navigator = el('div', { class: 'verlaufsnavigator' }, [
    el('h4', {}, ['Verlauf']),
    el('div', { class: 'verlaufwrap' }),
    el('p', { class: 'hint' }, [VERLAUF_ERKLAERUNG]),
  ]);

  const buehne = el('div', { class: 'linsenbuehne' });
  const kopf = el('div', { class: 'row spread linsenkopf' }, [
    el('h4', { class: 'linsentitel' }, []),
  ]);
  const schalter = el('div', { class: 'linsenwahl', role: 'tablist' });
  kopf.append(schalter);
  const flaeche = el('div', { class: 'graphwrap' });
  const erklaerung = el('p', { class: 'hint linsen-text' }, []);
  const beine = el('div', { class: 'linsen-fuss' }, [legend(), erklaerung]);
  const auswahlnote = el('p', { class: 'hint selection-note' }, []);
  auswahlnote.hidden = true;
  buehne.append(kopf, flaeche, auswahlnote, beine);

  for (const linse of LINSEN) {
    const knopf = el('button', {
      type: 'button', role: 'tab', 'data-linse': linse.id, title: linse.erklaerung,
    }, [linse.name]);
    knopf.addEventListener('click', () => {
      state.linse = linse.id;
      try {
        localStorage.setItem(LINSEN_KEY, linse.id);
      } catch {
        /* dann gilt die Wahl nur für diesen Besuch */
      }
      zeichneLinsen();
    });
    schalter.append(knopf);
  }

  raum.append(navigator, buehne);
  panel.append(raum);
  return panel;
}

/**
 * Zeichnet Verlauf und gewählte Linse neu.
 *
 * Eine Stelle, ein Bild: die Linsenfläche gibt es genau einmal, darum genügt
 * hier ein Aufruf — es gibt keine Funkenblöcke mehr, die einzeln nachziehen
 * müssten.
 */
function zeichneLinsen(): void {
  const panel = root.querySelector<HTMLElement>('.linsenraum');
  if (!panel) return;
  const bundle = state.bundle;
  if (!bundle) return;

  const eintrag = gewaehlterEintrag();
  const sparkId = eintrag?.spark.id ?? null;

  // --- Der Verlauf, links.
  const verlaufwrap = panel.querySelector<HTMLElement>('.verlaufwrap');
  if (verlaufwrap) {
    verlaufwrap.replaceChildren(
      renderVerlauf(bundle, sparkId, (id) => {
        state.gewaehlterSpark = id;
        zeichneLinsen();
        const ziel = blocks.get(id);
        ziel?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        ziel?.classList.add('angesteuert');
        window.setTimeout(() => ziel?.classList.remove('angesteuert'), 1600);
      }),
    );
  }

  // --- Die gewählte Linse, rechts.
  const flaeche = panel.querySelector<HTMLElement>('.graphwrap');
  const erklaerung = panel.querySelector<HTMLElement>('.linsen-text');
  const titel = panel.querySelector<HTMLElement>('.linsentitel');
  const beine = panel.querySelector<HTMLElement>('.linsen-fuss');
  if (!flaeche || !erklaerung || !titel || !beine) return;

  const art = state.linse;
  flaeche.replaceChildren();
  titel.textContent = eintrag
    ? `${LINSEN.find((l) => l.id === art)?.name ?? ''} — Funke ${eintrag.spark.seq}`
    : (LINSEN.find((l) => l.id === art)?.name ?? '');

  if (art === 'herkunft') {
    // Bezüge gelten sitzungsweit, nicht je Funke: sie sind der einzige Ort,
    // an dem steht, worauf du dich berufen kannst.
    titel.textContent = 'Herkunft — ganze Sitzung';
    flaeche.append(
      renderHerkunft(bundle.relations ?? [], beitragsNamen(), (auswahl) =>
        openAnswers(auswahl)),
    );
  } else if (art === 'szenario') {
    const szenario =
      (bundle.szenarien ?? []).find((s) => s.spark_id === sparkId)
      ?? (bundle.szenarien ?? [])[bundle.szenarien.length - 1]
      ?? null;
    if (szenario) titel.textContent = `Folgen — Funke ${szenario.seq}`;
    flaeche.append(
      renderSzenario(szenario, (auswahl, folge) =>
        openAnswers(auswahl, folgenMarker(folge))),
    );
    if (szenario?.ausgang) {
      flaeche.append(
        el('p', { class: 'hint ausgangstext' }, [
          `Ausgangsaussage — ${szenario.ausgang.label}: „${szenario.ausgang.auszug}"`,
        ]),
      );
    }
  } else if (art === 'zeit') {
    flaeche.append(renderZeit(eintrag?.jobs ?? [], (auswahl) => openAnswers(auswahl)));
  } else if (art === 'themen' && eintrag?.summary) {
    flaeche.append(
      renderThemen(eintrag.summary, eintrag.markers, (auswahl) => openAnswers(auswahl)),
    );
  } else if (eintrag?.summary) {
    flaeche.append(renderGraph(eintrag.summary, (auswahl) => openAnswers(auswahl)));
  } else {
    flaeche.append(
      el('p', { class: 'hint' }, ['Noch keine ausgewertete Antwort für diesen Funken.']),
    );
  }

  erklaerung.textContent = LINSEN.find((l) => l.id === art)?.erklaerung ?? '';
  // Die Legende erklärt die drei Bedeutungen. In der Zeitlinse sagen sie
  // nichts — dort steht keine Farbe für einen Befund.
  beine.querySelector('.legend')?.toggleAttribute('hidden', art === 'zeit');

  for (const knopf of panel.querySelectorAll('button[data-linse]')) {
    const gewaehlt = knopf.getAttribute('data-linse') === art;
    knopf.classList.toggle('gewaehlt', gewaehlt);
    knopf.setAttribute('aria-selected', String(gewaehlt));
  }
}

/**
 * Macht aus einer genannten Folge eine Auflage über den Antworttext.
 *
 * Die Marker entstehen hier nur für diesen einen Klick; gespeichert wird
 * nichts. Gelegt werden sie wie jede andere Auflage ausschließlich über
 * markers.ts — der Originaltext bleibt unangetastet, und ein Zitat, das nicht
 * mehr passt, wird dort stillschweigend übergangen statt den Text zu
 * verfälschen.
 */
function folgenMarker(folge: Folge): Marker[] {
  const art: Marker['kind'] =
    folge.gegensatz.length > 0
      ? 'widerspruch'
      : folge.anzahl > 1 ? 'uebereinstimmung' : 'einzigartig';
  const sitzung = state.bundle?.session.id ?? '';
  return folge.nennungen.map((nennung, i) => ({
    id: `folge-${folge.id}-${i}`,
    session_id: sitzung,
    spark_id: '',
    job_id: nennung.job_id,
    related_job_id: null,
    kind: art,
    start_offset: nennung.start_offset,
    end_offset: nennung.end_offset,
    quote: nennung.quote,
    note: `Genannte Folge — von ${folge.anzahl} von ${folge.von} Stimmen genannt`,
    topics: folge.themen,
  }));
}

/**
 * Öffnet genau die Antworten, die zum angeklickten Element gehören.
 *
 * Die Linsen stehen jetzt außerhalb der Funkenblöcke, und die Herkunfts-Linse
 * zeigt über Funken hinweg. Gesucht wird darum in der ganzen Seite, nicht mehr
 * in einem Block — sonst bliebe ein Bezug zwischen zwei Funken stumm.
 *
 * ``markers`` legt zusätzlich eine Fundstelle über den Text. Sie wird nicht
 * gespeichert und verschwindet mit der nächsten Auswahl wieder.
 */
function openAnswers(selection: GraphSelection, markers: Marker[] = []): void {
  root.querySelectorAll<HTMLElement>('.card.highlight')
    .forEach((card) => card.classList.remove('highlight'));

  // Die Auflage der vorigen Auswahl zuerst abräumen, damit nicht zwei
  // Fundstellen gleichzeitig behauptet werden.
  const vorher = [...state.zusatzMarker.keys()];
  state.zusatzMarker.clear();
  for (const jobId of markers.map((m) => m.job_id)) {
    state.zusatzMarker.set(jobId, markers.filter((m) => m.job_id === jobId));
  }
  for (const jobId of new Set([...vorher, ...state.zusatzMarker.keys()])) {
    zeichneKarteNeu(jobId);
  }

  let first: HTMLElement | null = null;
  for (const jobId of selection.jobIds) {
    const card = root.querySelector<HTMLElement>(`.card[data-job-id="${cssEscape(jobId)}"]`);
    if (!card) continue;
    card.classList.add('highlight');
    if (!first) first = card;
  }
  first?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const note = root.querySelector<HTMLElement>('.linsenraum .selection-note');
  if (note) {
    note.hidden = false;
    note.textContent = first
      ? `Geöffnet: ${selection.label}`
      : `${selection.label} — die zugehörige Antwort steht nicht mehr in dieser Ansicht.`;
  }
}

/** Alle Fundstellen einer Karte: die gespeicherten und die der aktuellen Auswahl. */
function markerFuer(entry: SparkEntry, jobId: string): Marker[] {
  return [...entry.markers, ...(state.zusatzMarker.get(jobId) ?? [])];
}

/** Zeichnet genau eine Karte neu — die übrigen bleiben unberührt. */
function zeichneKarteNeu(jobId: string): void {
  const entry = findEntry((e) => e.jobs.some((j) => j.id === jobId));
  const job = entry?.jobs.find((j) => j.id === jobId);
  if (!entry || !job) return;
  const card = root.querySelector<HTMLElement>(`.card[data-job-id="${cssEscape(jobId)}"]`);
  if (!card) return;
  updateCard(card, job, markerFuer(entry, jobId));
  ruesteKarteAus(card, job, entry);
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
  // Der Verlauf ist sitzungsweit: ein neuer Beitrag ist ein neuer Zweig.
  zeichneLinsen();
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
      // Der gemessene Beginn reist mit dem Ereignis; ohne ihn bliebe die
      // Zeitlinse bis zum nächsten vollständigen Laden blind.
      patchJob(String(payload.job_id), (job) => ({
        ...job,
        status: 'running',
        started_at: zeitpunkt(payload.started_at) ?? job.started_at,
      }));
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
        finished_at: zeitpunkt(payload.finished_at) ?? job.finished_at,
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
        finished_at: zeitpunkt(payload.finished_at) ?? job.finished_at,
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
        finished_at: zeitpunkt(payload.finished_at) ?? job.finished_at,
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
      // Die Konsequenzkarte kommt vom Server mit — gerechnet wird sie dort,
      // damit Satzzerlegung und Ähnlichkeit genau einmal existieren.
      if (payload.szenario) {
        const karte = payload.szenario as Szenario;
        bundle.szenarien = [
          ...(bundle.szenarien ?? []).filter((x) => x.spark_id !== karte.spark_id),
          karte,
        ].sort((x, y) => x.seq - y.seq);
      }
      const block = blocks.get(sparkId);
      if (block) {
        for (const job of entry.jobs) {
          const card = block.querySelector<HTMLElement>(
            `.card[data-job-id="${cssEscape(job.id)}"]`,
          );
          if (card) {
            updateCard(card, job, markerFuer(entry, job.id));
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

/**
 * Ein gemessener Zeitpunkt aus einem Ereignis — oder null.
 *
 * Nur was als Zahl ankommt, gilt als gemessen. Es wird nichts ersatzweise aus
 * der Ankunftszeit des Ereignisses gebildet: das wäre eine erfundene Messung.
 */
function zeitpunkt(wert: unknown): number | null {
  return typeof wert === 'number' && Number.isFinite(wert) ? wert : null;
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
    updateCard(card, entry.jobs[index], markerFuer(entry, jobId));
    ruesteKarteAus(card, entry.jobs[index], entry);
  }
  // Die Zeitlinse liest dieselben Zeitstempel — sie muss mitziehen.
  if (state.linse === 'zeit') zeichneLinsen();
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
