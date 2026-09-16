import { api } from './api';
import { el } from './render';
import type { AdminSettings, CustomHint, ProviderRow } from './types';

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

type Melder = (text: string, fehler?: boolean) => void;

/**
 * Die Einstellungsseite.
 *
 * Anbieter sind hier Daten: die mitgelieferten lassen sich ändern und
 * abschalten, beliebige weitere kommen über „Eigenen Anbieter eintragen“ dazu.
 * Schlüssel werden nur gesendet, nie zurückgelesen — der Server gibt allein
 * Herkunft und die letzten vier Zeichen heraus.
 */
export function renderSettings(settingsAvailable: boolean, onClose: () => void): HTMLElement {
  const panel = el('section', { class: 'flaeche settings', id: 'einstellungen' }, [
    el('div', { class: 'row spread' }, [
      el('h4', {}, ['Einstellungen — Anbieter am Tisch']),
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
          'Alternativ lassen sich die Schlüssel weiterhin direkt als Umgebungsvariablen ' +
          'setzen — die Namen stehen in der README.',
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

  const melde: Melder = (text, fehler = false) => {
    meldung.className = fehler ? 'hint error' : 'hint';
    meldung.textContent = text;
  };

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
    const neu = (d: AdminSettings) => zeichne(d, token);
    inhalt.replaceChildren();

    if (!daten.editable) {
      inhalt.append(
        el('p', { class: 'hint' }, [
          'Im Test- und Entwicklungsbetrieb steht der Tisch fest im Quelltext; ' +
            'Anbieter lassen sich hier nicht ändern.',
        ]),
      );
    }

    const bereit = daten.providers.filter((p) => p.ready).length;
    inhalt.append(
      el('p', { class: 'hint' }, [
        `${bereit} von ${daten.providers.length} Anbietern einsatzbereit. ` +
          'Ein Anbieter kommt an den Tisch, wenn er eingeschaltet ist und einen Schlüssel hat.',
      ]),
    );

    for (const zeile of daten.providers) {
      inhalt.append(anbieterBlock(zeile, token, daten.editable, melde, neu));
    }

    if (daten.editable) {
      inhalt.append(neuerAnbieter(daten, token, melde, neu));
    }
    inhalt.append(zeitgrenze(daten, token, melde, neu));
    inhalt.append(kuratorwahl(daten, token, melde, neu));
    inhalt.append(
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

// ------------------------------------------------------------- Ein Anbieter

/** Preis als Text fürs Eingabefeld — unbekannt bleibt leer, nicht null. */
function preisText(wert: number | null): string {
  return wert === null || wert === undefined ? '' : String(wert);
}

function anbieterBlock(
  zeile: ProviderRow,
  token: string,
  editable: boolean,
  melde: Melder,
  neuZeichnen: (daten: AdminSettings) => void,
): HTMLElement {
  const block = el('div', {
    class: `anbieter${zeile.ready ? ' bereit' : ''}${zeile.enabled ? '' : ' aus'}`,
    'data-provider': zeile.id,
  });

  const schalter = el('input', {
    type: 'checkbox',
    id: `an-${zeile.id}`,
    'aria-label': `${zeile.label} am Tisch`,
  }) as HTMLInputElement;
  schalter.checked = zeile.enabled;
  schalter.disabled = !editable;
  schalter.addEventListener('change', async () => {
    try {
      neuZeichnen(await api.updateProvider(token, zeile.id, { enabled: schalter.checked }));
      melde(`${zeile.label}: ${schalter.checked ? 'am Tisch' : 'abgeschaltet'}.`);
    } catch (fehler) {
      schalter.checked = !schalter.checked;
      melde((fehler as Error).message, true);
    }
  });

  block.append(
    el('div', { class: 'anbieter-kopf' }, [
      el('div', { class: 'row' }, [
        el('label', { class: 'schalter', for: `an-${zeile.id}` }, [schalter, zeile.label]),
      ]),
      el('div', { class: 'tags' }, [
        el('span', { class: `tag ${zeile.ready ? 'done' : 'error'}` }, [
          zeile.ready ? 'bereit' : 'nicht bereit',
        ]),
        el('span', { class: 'tag' }, [
          !zeile.needs_key
            ? 'ohne Schlüssel'
            : zeile.key_hint
              ? `Schlüssel ${zeile.key_hint} · ${QUELLE_TEXT[zeile.key_source] ?? zeile.key_source}`
              : 'kein Schlüssel',
        ]),
        el('span', { class: 'tag' }, [zeile.is_preset ? 'mitgeliefert' : 'selbst eingetragen']),
      ]),
    ]),
  );

  if (!zeile.ready && zeile.reason) {
    block.append(el('p', { class: 'hint error' }, [zeile.reason]));
  }

  const felder = el('div', { class: 'anbieter-felder' });
  const eingaben: Record<string, HTMLInputElement> = {};

  const feld = (
    name: string,
    beschriftung: string,
    wert: string,
    typ = 'text',
    platzhalter = '',
  ) => {
    const eingabe = el('input', {
      type: typ,
      id: `${name}-${zeile.id}`,
      placeholder: platzhalter,
      autocomplete: typ === 'password' ? 'off' : 'on',
    }) as HTMLInputElement;
    eingabe.value = wert;
    eingabe.disabled = !editable;
    eingaben[name] = eingabe;
    felder.append(
      el('div', { class: 'feld' }, [
        el('label', { for: `${name}-${zeile.id}` }, [beschriftung]),
        eingabe,
      ]),
    );
  };

  if (zeile.needs_key) {
    feld(
      'key',
      'Schlüssel',
      '',
      'password',
      zeile.key_hint ? 'Neuer Schlüssel (bleibt sonst unverändert)' : 'Schlüssel eintragen',
    );
  }
  feld('model', 'Modellname', zeile.model);
  if (zeile.kind === 'openai' || zeile.kind === 'google' || !zeile.is_preset) {
    feld('base', 'Basis-Adresse', zeile.base_url ?? '', 'text', 'Standard des Anbieters');
  }
  // Ohne eingetragenen Preis wird nichts geschätzt: die Kosten bleiben dann
  // ausdrücklich unbekannt, statt eine Zahl zu erfinden.
  feld(
    'price_in', 'Preis Eingabe je Mio. Token', preisText(zeile.price_in),
    'number', 'leer = unbekannt',
  );
  feld(
    'price_out', 'Preis Ausgabe je Mio. Token', preisText(zeile.price_out),
    'number', 'leer = unbekannt',
  );
  block.append(felder);
  block.append(
    el('p', { class: 'hint' }, [
      'Die Preise trägst du selbst ein, in deiner Währung. Fehlt einer, steht auf der ' +
        'Karte „Kosten unbekannt“ — es wird nichts geraten.',
    ]),
  );

  const knoepfe = el('div', { class: 'row' });

  const speichern = el('button', { class: 'primary', type: 'button' }, ['Speichern']);
  speichern.disabled = !editable;
  speichern.addEventListener('click', async () => {
    const patch: Record<string, string | boolean | number> = {};
    if (eingaben.key?.value.trim()) {
      patch.api_key = eingaben.key.value.trim();
      // Wer einen Schlüssel einträgt, will den Anbieter am Tisch haben.
      if (!zeile.enabled) patch.enabled = true;
    }
    if (eingaben.model && eingaben.model.value.trim() !== zeile.model) {
      patch.model = eingaben.model.value.trim();
    }
    if (eingaben.base && eingaben.base.value.trim() !== (zeile.base_url ?? '')) {
      patch.base_url = eingaben.base.value.trim();
    }
    for (const name of ['price_in', 'price_out'] as const) {
      const eingabe = eingaben[name];
      if (!eingabe) continue;
      const roh = eingabe.value.trim();
      if (roh === preisText(zeile[name])) continue;
      // Leeres Feld heißt: Preis wieder unbekannt. Die Null löscht ihn serverseitig.
      const zahl = roh === '' ? 0 : Number(roh);
      if (!Number.isFinite(zahl) || zahl < 0) {
        melde('Preise müssen Zahlen ab null sein.', true);
        return;
      }
      patch[name] = zahl;
    }
    if (Object.keys(patch).length === 0) {
      melde('Nichts zu speichern.', true);
      return;
    }
    try {
      neuZeichnen(await api.updateProvider(token, zeile.id, patch));
      melde(
        patch.enabled === true
          ? `${zeile.label}: gespeichert und an den Tisch geholt.`
          : `${zeile.label}: gespeichert.`,
      );
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });
  knoepfe.append(speichern);

  const pruefen = el('button', { type: 'button' }, ['Prüfen (echter Aufruf)']);
  pruefen.addEventListener('click', async () => {
    pruefen.setAttribute('disabled', 'true');
    melde(`${zeile.label} wird geprüft …`);
    try {
      const ergebnis = await api.testProvider(token, zeile.id);
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
  knoepfe.append(pruefen);

  if (zeile.needs_key && zeile.key_source === 'einstellungen' && editable) {
    const entfernen = el('button', { type: 'button' }, ['Schlüssel entfernen']);
    entfernen.addEventListener('click', async () => {
      try {
        neuZeichnen(await api.updateProvider(token, zeile.id, { api_key: '' }));
        melde(`${zeile.label}: Schlüssel entfernt.`);
      } catch (fehler) {
        melde((fehler as Error).message, true);
      }
    });
    knoepfe.append(entfernen);
  }

  if (!zeile.is_preset && editable) {
    const loeschen = el('button', { type: 'button', class: 'gefahr' }, ['Anbieter löschen']);
    loeschen.addEventListener('click', async () => {
      try {
        neuZeichnen(await api.deleteProvider(token, zeile.id));
        melde(`${zeile.label}: gelöscht.`);
      } catch (fehler) {
        melde((fehler as Error).message, true);
      }
    });
    knoepfe.append(loeschen);
  }

  if (zeile.key_url) {
    knoepfe.append(
      el('a', { class: 'btn', href: zeile.key_url, target: '_blank', rel: 'noreferrer noopener' },
        ['Schlüssel holen ↗']),
    );
  }
  if (zeile.models_url) {
    knoepfe.append(
      el('a', { class: 'btn', href: zeile.models_url, target: '_blank', rel: 'noreferrer noopener' },
        ['Modellliste ↗']),
    );
  }

  block.append(knoepfe);
  return block;
}

// ------------------------------------------------------- Eigener Anbieter

function neuerAnbieter(
  daten: AdminSettings,
  token: string,
  melde: Melder,
  neuZeichnen: (daten: AdminSettings) => void,
): HTMLElement {
  const details = el('details', { id: 'neuer-anbieter' });
  details.append(el('summary', {}, ['Eigenen Anbieter eintragen']));
  details.append(
    el('p', { class: 'hint' }, [
      'Alles, was die OpenAI-Schnittstelle spricht, lässt sich hier anschließen — ' +
        'OpenRouter, Together, Fireworks, Groq oder ein Server im eigenen Netz.',
    ]),
  );

  const felder = el('div', { class: 'anbieter-felder' });
  const mach = (id: string, beschriftung: string, platzhalter: string, typ = 'text') => {
    const eingabe = el('input', { type: typ, id, placeholder: platzhalter }) as HTMLInputElement;
    felder.append(
      el('div', { class: 'feld' }, [el('label', { for: id }, [beschriftung]), eingabe]),
    );
    return eingabe;
  };

  const name = mach('neu-label', 'Anzeigename', 'z. B. Groq');
  const art = el('select', { id: 'neu-kind', 'aria-label': 'Art der Schnittstelle' }) as HTMLSelectElement;
  for (const k of daten.kinds) {
    const option = document.createElement('option');
    option.value = k.id;
    option.textContent = k.label;
    art.append(option);
  }
  felder.append(
    el('div', { class: 'feld' }, [
      el('label', { for: 'neu-kind' }, ['Art der Schnittstelle']),
      art,
    ]),
  );
  const adresse = mach('neu-base', 'Basis-Adresse', 'https://api.example.com/v1');
  const modell = mach('neu-model', 'Modellname', 'z. B. llama-3.3-70b-versatile');
  const schluessel = mach('neu-key', 'Schlüssel', 'Schlüssel eintragen', 'password');

  details.append(felder);

  const vorschlaege = el('div', { class: 'vorschlaege' });
  for (const hinweis of daten.custom_hints as CustomHint[]) {
    const knopf = el('button', { type: 'button' }, [hinweis.label]);
    knopf.addEventListener('click', () => {
      name.value = hinweis.label;
      art.value = 'openai';
      adresse.value = hinweis.base_url;
      modell.value = hinweis.model;
      schluessel.focus();
    });
    vorschlaege.append(knopf);
  }
  details.append(
    el('div', {}, [el('p', { class: 'hint' }, ['Vorlagen zum Ausfüllen:']), vorschlaege]),
  );

  const anlegen = el('button', { class: 'primary', type: 'button', id: 'neu-anlegen' },
    ['Anbieter hinzufügen']);
  anlegen.addEventListener('click', async () => {
    if (!name.value.trim() || !modell.value.trim()) {
      melde('Anzeigename und Modellname werden gebraucht.', true);
      return;
    }
    try {
      const antwort = await api.createProvider(token, {
        label: name.value.trim(),
        kind: art.value,
        base_url: adresse.value.trim(),
        model: modell.value.trim(),
        api_key: schluessel.value.trim(),
      });
      melde(`${name.value.trim()} hinzugefügt.`);
      neuZeichnen(antwort);
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });
  details.append(el('div', { class: 'row' }, [anlegen]));
  return details;
}

// ----------------------------------------------------------------- Zeitgrenze

function zeitgrenze(
  daten: AdminSettings,
  token: string,
  melde: Melder,
  neuZeichnen: (daten: AdminSettings) => void,
): HTMLElement {
  const eingabe = el('input', {
    type: 'number', min: '5', max: '600', id: 'timeout',
    'aria-label': 'Zeitgrenze je Modellaufruf in Sekunden',
  }) as HTMLInputElement;
  eingabe.value = String(daten.request_timeout_s);

  const knopf = el('button', { type: 'button' }, ['Zeitgrenze speichern']);
  knopf.addEventListener('click', async () => {
    try {
      neuZeichnen(await api.saveTimeout(token, Number(eingabe.value)));
      melde('Zeitgrenze gespeichert.');
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });

  return el('div', { class: 'anbieter' }, [
    el('div', { class: 'feld' }, [
      el('label', { for: 'timeout' }, ['Zeitgrenze je Modellaufruf (Sekunden)']),
      el('div', { class: 'row' }, [eingabe, knopf]),
    ]),
    el('p', { class: 'hint' }, [
      `Derzeit ${QUELLE_TEXT[daten.timeout_source] ?? daten.timeout_source}.`,
    ]),
  ]);
}

// -------------------------------------------------------------------- Kurator

/** Wer die Antworten zusammenfasst — freiwillig, und niemals als Ersatz.
 *
 * Die Kuratierung erscheint als eigener Beitrag am Tisch. Die
 * Originalantworten bleiben unverändert stehen.
 */
function kuratorwahl(
  daten: AdminSettings,
  token: string,
  melde: Melder,
  neuZeichnen: (daten: AdminSettings) => void,
): HTMLElement {
  const auswahl = el('select', {
    id: 'kurator', 'aria-label': 'Anbieter für die Kuratierung',
  }) as HTMLSelectElement;
  const leer = document.createElement('option');
  leer.value = '';
  leer.textContent = '— keiner, Kuratierung abgeschaltet —';
  auswahl.append(leer);
  for (const zeile of daten.providers.filter((z) => z.enabled && z.ready)) {
    const option = document.createElement('option');
    option.value = zeile.id;
    option.textContent = `${zeile.label} · ${zeile.model}`;
    auswahl.append(option);
  }
  auswahl.value = daten.curator;

  const knopf = el('button', { type: 'button', id: 'kurator-speichern' }, ['Kurator setzen']);
  knopf.addEventListener('click', async () => {
    try {
      neuZeichnen(await api.saveCurator(token, auswahl.value));
      melde(auswahl.value ? 'Kurator gesetzt.' : 'Kuratierung abgeschaltet.');
    } catch (fehler) {
      melde((fehler as Error).message, true);
    }
  });

  return el('div', { class: 'anbieter' }, [
    el('div', { class: 'feld' }, [
      el('label', { for: 'kurator' }, ['Kuratierung durch']),
      el('div', { class: 'row' }, [auswahl, knopf]),
    ]),
    el('p', { class: 'hint' }, [
      'Auf Wunsch fasst dieses Modell die Antworten eines Funkens zusammen. Die ' +
        'Zusammenfassung ist ein eigener Beitrag und ersetzt keine Originalantwort. ' +
        'Ohne Auswahl gibt es keine Kuratierung.',
    ]),
  ]);
}
