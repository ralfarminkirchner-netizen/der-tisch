import { api } from './api';
import { el } from './render';
import type { AdminSettings, AdminProviderRow } from './types';

const TOKEN_KEY = 'xpertentisch.admin';

/** Das Zugangswort bleibt im Sitzungsspeicher des Browsers — nie im Bundle. */
export function readToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

function writeToken(value: string): void {
  try {
    if (value) sessionStorage.setItem(TOKEN_KEY, value);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* Privater Modus: das Zugangswort gilt dann nur für diese Ansicht. */
  }
}

const QUELLE_TEXT: Record<string, string> = {
  einstellungen: 'hier hinterlegt',
  umgebung: 'aus der Umgebung',
  fehlt: 'nicht hinterlegt',
};

/**
 * Die Einstellungsseite.
 *
 * Schlüssel werden nur gesendet, nie zurückgelesen: der Server gibt
 * ausschließlich Herkunft und die letzten vier Zeichen heraus.
 */
export function renderSettings(settingsAvailable: boolean, onClose: () => void): HTMLElement {
  const panel = el('section', { class: 'glass settings', id: 'einstellungen' }, [
    el('div', { class: 'row spread' }, [
      el('h4', {}, ['Einstellungen']),
      (() => {
        const zu = el('button', { type: 'button' }, ['Schließen']);
        zu.addEventListener('click', onClose);
        return zu;
      })(),
    ]),
  ]);

  if (!settingsAvailable) {
    panel.append(
      el('p', { class: 'hint' }, [
        'Die Einstellungen sind gesperrt, weil kein Zugangswort eingerichtet ist. ' +
          'So schaltest du sie frei:',
      ]),
      el('ol', { class: 'steps' }, [
        el('li', {}, ['Beim Betreiber (z. B. Railway) eine Variable XT_ADMIN_TOKEN anlegen.']),
        el('li', {}, ['Als Wert ein langes, selbst gewähltes Wort eintragen.']),
        el('li', {}, ['Den Dienst neu starten und diese Seite erneut öffnen.']),
      ]),
      el('p', { class: 'hint' }, [
        'Ohne diesen Schutz könnte jede Person mit dem Link die Zugangsdaten ändern. ' +
          'Alternativ lassen sich OPENAI_API_KEY und ANTHROPIC_API_KEY weiterhin direkt ' +
          'als Umgebungsvariablen setzen.',
      ]),
    );
    return panel;
  }

  const meldung = el('p', { class: 'hint', id: 'einstellungen-meldung' }, []);
  const inhalt = el('div', { class: 'settings-body' });

  const tokenFeld = el('input', {
    type: 'password',
    id: 'admin-token',
    placeholder: 'Zugangswort',
    'aria-label': 'Zugangswort für die Einstellungen',
    autocomplete: 'current-password',
  }) as HTMLInputElement;
  tokenFeld.value = readToken();

  const oeffnen = el('button', { class: 'primary', type: 'button' }, ['Öffnen']);
  const tokenZeile = el('div', { class: 'row' }, [tokenFeld, oeffnen]);

  panel.append(tokenZeile, meldung, inhalt);

  function melde(text: string, fehler = false): void {
    meldung.className = fehler ? 'hint error' : 'hint';
    meldung.textContent = text;
  }

  async function laden(): Promise<void> {
    const token = tokenFeld.value.trim();
    if (!token) {
      melde('Bitte das Zugangswort eingeben.', true);
      return;
    }
    oeffnen.setAttribute('disabled', 'true');
    try {
      const daten = await api.adminSettings(token);
      writeToken(token);
      tokenZeile.hidden = true;
      melde('');
      zeichne(daten, token);
    } catch (fehler) {
      writeToken('');
      melde((fehler as Error).message, true);
    } finally {
      oeffnen.removeAttribute('disabled');
    }
  }

  function zeichne(daten: AdminSettings, token: string): void {
    inhalt.replaceChildren();
    for (const zeile of daten.providers) {
      inhalt.append(providerZeile(zeile, token, melde, (neu) => zeichne(neu, token)));
    }

    const zeitFeld = el('input', {
      type: 'number', min: '5', max: '600', id: 'timeout',
      'aria-label': 'Zeitgrenze je Modellaufruf in Sekunden',
    }) as HTMLInputElement;
    zeitFeld.value = String(daten.request_timeout_s);

    const zeitKnopf = el('button', { type: 'button' }, ['Zeitgrenze speichern']);
    zeitKnopf.addEventListener('click', async () => {
      try {
        const neu = await api.saveAdminSettings(token, {
          request_timeout_s: Number(zeitFeld.value),
        });
        melde('Zeitgrenze gespeichert.');
        zeichne(neu, token);
      } catch (fehler) {
        melde((fehler as Error).message, true);
      }
    });

    inhalt.append(
      el('div', { class: 'settings-zeile' }, [
        el('label', { for: 'timeout' }, ['Zeitgrenze je Modellaufruf (Sekunden)']),
        el('div', { class: 'row' }, [zeitFeld, zeitKnopf]),
        el('p', { class: 'hint' }, [`Derzeit ${QUELLE_TEXT[daten.timeout_source] ?? daten.timeout_source}.`]),
      ]),
      el('p', { class: 'hint' }, [
        'Hinterlegte Schlüssel liegen in der Datenbank des Dienstes und werden nie ' +
          'an die Oberfläche zurückgegeben — sichtbar sind nur Herkunft und die letzten ' +
          'vier Zeichen.',
      ]),
    );
  }

  oeffnen.addEventListener('click', () => void laden());
  tokenFeld.addEventListener('keydown', (ereignis) => {
    if ((ereignis as KeyboardEvent).key === 'Enter') {
      ereignis.preventDefault();
      void laden();
    }
  });
  if (tokenFeld.value) void laden();

  return panel;
}

function providerZeile(
  zeile: AdminProviderRow,
  token: string,
  melde: (text: string, fehler?: boolean) => void,
  neuZeichnen: (daten: AdminSettings) => void,
): HTMLElement {
  const block = el('div', { class: 'settings-zeile', 'data-provider': zeile.provider });

  const zustand = el('span', { class: `tag ${zeile.ready ? 'done' : 'error'}` }, [
    zeile.ready ? 'bereit' : 'nicht bereit',
  ]);
  block.append(
    el('div', { class: 'row spread' }, [
      el('strong', {}, [zeile.label]),
      el('div', { class: 'tags' }, [
        zustand,
        el('span', { class: 'tag' }, [
          !zeile.needs_key
            ? 'ohne Schlüssel'
            : zeile.key_hint
              ? `Schlüssel ${zeile.key_hint} · ${QUELLE_TEXT[zeile.key_source] ?? zeile.key_source}`
              : 'kein Schlüssel',
        ]),
      ]),
    ]),
  );
  if (!zeile.ready && zeile.reason) {
    block.append(el('p', { class: 'hint error' }, [zeile.reason]));
  }

  const schluesselFeld = el('input', {
    type: 'password',
    id: `key-${zeile.provider}`,
    placeholder: zeile.key_hint ? 'Neuen Schlüssel eintragen (bleibt sonst unverändert)' : 'Schlüssel eintragen',
    'aria-label': `Schlüssel für ${zeile.label}`,
    autocomplete: 'off',
  }) as HTMLInputElement;

  const modellFeld = el('input', {
    type: 'text',
    id: `model-${zeile.provider}`,
    'aria-label': `Modellname für ${zeile.label}`,
  }) as HTMLInputElement;
  modellFeld.value = zeile.model;

  const speichern = el('button', { class: 'primary', type: 'button' }, ['Speichern']);
  const pruefen = el('button', { type: 'button' }, ['Prüfen (echter Aufruf)']);
  const entfernen = el('button', { type: 'button' }, ['Schlüssel entfernen']);
  if (zeile.key_source !== 'einstellungen') entfernen.setAttribute('disabled', 'true');

  speichern.addEventListener('click', async () => {
    const aenderung: Record<string, string> = {};
    if (schluesselFeld.value.trim()) aenderung[zeile.key_field] = schluesselFeld.value.trim();
    if (modellFeld.value.trim() && modellFeld.value.trim() !== zeile.model) {
      aenderung[zeile.model_field] = modellFeld.value.trim();
    }
    if (Object.keys(aenderung).length === 0) {
      melde('Nichts zu speichern.', true);
      return;
    }
    try {
      const neu = await api.saveAdminSettings(token, aenderung);
      schluesselFeld.value = '';
      melde(`${zeile.label}: gespeichert.`);
      neuZeichnen(neu);
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });

  entfernen.addEventListener('click', async () => {
    try {
      const neu = await api.saveAdminSettings(token, { [zeile.key_field]: '' });
      melde(`${zeile.label}: Schlüssel entfernt.`);
      neuZeichnen(neu);
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });

  pruefen.addEventListener('click', async () => {
    pruefen.setAttribute('disabled', 'true');
    melde(`${zeile.label} wird geprüft …`);
    try {
      const ergebnis = await api.testProvider(token, zeile.provider);
      melde(
        ergebnis.ok
          ? `${zeile.label} antwortet (${ergebnis.latency_ms} ms): ${ergebnis.detail}`
          : `${zeile.label}: ${ergebnis.detail}`,
        !ergebnis.ok,
      );
    } catch (fehler) {
      melde((fehler as Error).message, true);
    } finally {
      pruefen.removeAttribute('disabled');
    }
  });

  if (zeile.needs_key) {
    block.append(el('label', { for: `key-${zeile.provider}` }, ['Schlüssel']), schluesselFeld);
  }
  block.append(
    el('label', { for: `model-${zeile.provider}` }, ['Modellname']),
    modellFeld,
    el('div', { class: 'row' }, zeile.needs_key ? [speichern, pruefen, entfernen] : [speichern, pruefen]),
  );
  return block;
}
