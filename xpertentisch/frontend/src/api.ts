import type { AppConfig, HealthInfo, Job, Session, SessionBundle, Spark } from './types';

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* Antwort ohne JSON-Körper */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthInfo>('/api/health'),
  config: () => request<AppConfig>('/api/config'),
  createSession: (title: string) =>
    request<Session>('/api/sessions', { method: 'POST', body: JSON.stringify({ title }) }),
  getSession: (id: string) => request<SessionBundle>(`/api/sessions/${id}`),
  closeSession: (id: string, note: string) =>
    request<Session>(`/api/sessions/${id}/close`, {
      method: 'POST',
      body: JSON.stringify({ note }),
    }),
  createSpark: (
    sessionId: string,
    prompt: string,
    clientRequestId: string,
    modelIds: string[] | null,
  ) =>
    request<{ spark: Spark; jobs: Job[]; duplicate: boolean }>(
      `/api/sessions/${sessionId}/sparks`,
      {
        method: 'POST',
        body: JSON.stringify({
          prompt,
          client_request_id: clientRequestId,
          model_ids: modelIds,
        }),
      },
    ),
};

/** Erzeugt eine stabile Kennung je Absendevorgang.
 *
 * Wird ein Absenden wiederholt (Netzfehler, doppelter Klick, erneutes Senden
 * nach Verbindungsabbruch), bleibt die Kennung gleich. Der Server erkennt die
 * Wiederholung und startet keine zweiten Modellaufrufe.
 */
export function newRequestId(): string {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && 'randomUUID' in cryptoObj) return `req-${cryptoObj.randomUUID()}`;
  return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export interface EventStreamHandlers {
  onEvent: (type: string, payload: unknown) => void;
  onOpen?: () => void;
  onError?: () => void;
}

const EVENT_TYPES = [
  'sitzung.angelegt',
  'sitzung.abgeschlossen',
  'funke.angelegt',
  'auftrag.laeuft',
  'auftrag.fertig',
  'auftrag.fehler',
  'auftrag.unterbrochen',
  'einschaetzung.fertig',
];

/** Verbindet den Ereignisstrom.
 *
 * EventSource sendet nach einem Abbruch selbsttätig die zuletzt empfangene
 * Ereigniskennung als `Last-Event-ID` mit; der Server liefert daraufhin nur
 * die verpassten Ereignisse nach. Es entstehen dabei keine neuen Aufträge.
 */
export function connectEvents(
  sessionId: string,
  lastEventId: number,
  handlers: EventStreamHandlers,
): () => void {
  const url = `/api/sessions/${sessionId}/events?last_event_id=${lastEventId}`;
  const source = new EventSource(url);
  source.onopen = () => handlers.onOpen?.();
  source.onerror = () => handlers.onError?.();
  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (event) => {
      const message = event as MessageEvent<string>;
      let payload: unknown = null;
      try {
        payload = JSON.parse(message.data);
      } catch {
        payload = null;
      }
      handlers.onEvent(type, payload);
    });
  }
  return () => source.close();
}
