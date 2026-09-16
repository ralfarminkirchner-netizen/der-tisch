import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { AdminSettings, ProviderRow } from '../src/types';

const adminSettings = vi.fn();
const createProvider = vi.fn();
const updateProvider = vi.fn();
const deleteProvider = vi.fn();
const testProvider = vi.fn();
const saveTimeout = vi.fn();

vi.mock('../src/api', () => ({
  api: { adminSettings, createProvider, updateProvider, deleteProvider, testProvider, saveTimeout },
}));

const { renderSettings } = await import('../src/settings');

function zeile(over: Partial<ProviderRow> & Pick<ProviderRow, 'id' | 'label'>): ProviderRow {
  return {
    kind: 'openai', base_url: null, model: 'ein-modell', enabled: true, is_preset: true,
    needs_key: true, key_source: 'einstellungen', key_hint: '…cdef', ready: true, reason: '',
    key_env: 'IRGENDEIN_API_KEY', key_url: 'https://beispiel.test/keys',
    models_url: 'https://beispiel.test/models',
    ...over,
  } as ProviderRow;
}

const ANTWORT: AdminSettings = {
  providers: [
    zeile({ id: 'openai', label: 'OpenAI' }),
    zeile({
      id: 'google', label: 'Google Gemini', kind: 'google', key_source: 'fehlt',
      key_hint: '', ready: false, enabled: false,
      reason: 'Kein Schlüssel hinterlegt — in den Einstellungen eintragen oder GOOGLE_API_KEY setzen.',
    }),
    zeile({
      id: 'groq', label: 'Groq', is_preset: false, key_source: 'umgebung',
      base_url: 'https://api.groq.com/openai/v1', key_url: '', models_url: '',
    }),
  ],
  kinds: [
    { id: 'openai', label: 'OpenAI-kompatibel' },
    { id: 'anthropic', label: 'Anthropic' },
    { id: 'google', label: 'Google' },
  ],
  custom_hints: [
    { label: 'Groq', base_url: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile' },
  ],
  request_timeout_s: 120,
  timeout_source: 'umgebung',
  env: 'production',
  fake_providers_enabled: false,
  editable: true,
};

const warte = () => new Promise<void>((fertig) => setTimeout(fertig, 0));

async function geoeffnet(daten: AdminSettings = ANTWORT): Promise<HTMLElement> {
  adminSettings.mockResolvedValue(daten);
  const panel = renderSettings(true, () => {});
  document.body.append(panel);
  panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
  panel.querySelector<HTMLButtonElement>('button.primary')!.click();
  await warte();
  return panel;
}

function knopf(panel: HTMLElement, provider: string, text: string): HTMLButtonElement {
  return [...panel.querySelectorAll<HTMLButtonElement>(`[data-provider="${provider}"] button`)]
    .find((b) => b.textContent?.startsWith(text))!;
}

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  document.body.replaceChildren();
});

describe('Ohne eingerichtetes Zugangswort', () => {
  it('erklärt den Weg statt ein Feld anzubieten', () => {
    const panel = renderSettings(false, () => {});
    expect(panel.querySelector('#admin-token')).toBeNull();
    expect(panel.textContent).toContain('XT_ADMIN_TOKEN');
    expect(panel.querySelectorAll('.steps li')).toHaveLength(3);
    expect(panel.textContent).toContain('jede Person mit dem Link');
  });
});

describe('Anbieterliste', () => {
  it('fragt zuerst das Zugangswort ab', () => {
    const panel = renderSettings(true, () => {});
    const feld = panel.querySelector<HTMLInputElement>('#admin-token')!;
    expect(feld.type).toBe('password');
    expect(adminSettings).not.toHaveBeenCalled();
  });

  it('zeigt alle Anbieter mit Zustand und Herkunft', async () => {
    const panel = await geoeffnet();
    expect(adminSettings).toHaveBeenCalledWith('zugangswort');
    expect(panel.querySelectorAll('.anbieter[data-provider]')).toHaveLength(3);
    expect(panel.textContent).toContain('2 von 3 Anbietern einsatzbereit');
    expect(panel.querySelector('[data-provider="openai"]')!.textContent).toContain('…cdef');
    expect(panel.querySelector('[data-provider="groq"]')!.textContent)
      .toContain('selbst eingetragen');
  });

  it('nennt den Grund, wenn ein Anbieter nicht bereit ist', async () => {
    const panel = await geoeffnet();
    const block = panel.querySelector('[data-provider="google"]')!;
    expect(block.textContent).toContain('nicht bereit');
    expect(block.textContent).toContain('Kein Schlüssel hinterlegt');
  });

  it('zeigt nie einen Schlüssel, nur die Erkennungshilfe', async () => {
    const panel = await geoeffnet();
    const feld = panel.querySelector<HTMLInputElement>('#key-openai')!;
    expect(feld.type).toBe('password');
    expect(feld.value).toBe('');
  });

  it('schaltet einen Anbieter an und ab', async () => {
    updateProvider.mockResolvedValue(ANTWORT);
    const panel = await geoeffnet();
    const schalter = panel.querySelector<HTMLInputElement>('#an-google')!;
    expect(schalter.checked).toBe(false);
    schalter.checked = true;
    schalter.dispatchEvent(new Event('change'));
    await warte();
    expect(updateProvider).toHaveBeenCalledWith('zugangswort', 'google', { enabled: true });
  });

  it('schickt einen neuen Schlüssel und leert das Feld', async () => {
    updateProvider.mockResolvedValue(ANTWORT);
    const panel = await geoeffnet();
    panel.querySelector<HTMLInputElement>('#key-openai')!.value = 'sk-neu-1234';
    knopf(panel, 'openai', 'Speichern').click();
    await warte();
    expect(updateProvider).toHaveBeenCalledWith('zugangswort', 'openai', { api_key: 'sk-neu-1234' });
    expect(panel.querySelector<HTMLInputElement>('#key-openai')!.value).toBe('');
  });

  it('holt einen Anbieter mit dem ersten Schlüssel gleich an den Tisch', async () => {
    updateProvider.mockResolvedValue(ANTWORT);
    const panel = await geoeffnet();
    panel.querySelector<HTMLInputElement>('#key-google')!.value = 'AIza-neu-5678';
    knopf(panel, 'google', 'Speichern').click();
    await warte();
    expect(updateProvider).toHaveBeenCalledWith('zugangswort', 'google',
      { api_key: 'AIza-neu-5678', enabled: true });
    expect(panel.querySelector('#einstellungen-meldung')!.textContent)
      .toContain('an den Tisch geholt');
  });

  it('lässt einen bereits eingeschalteten Anbieter in Ruhe', async () => {
    updateProvider.mockResolvedValue(ANTWORT);
    const panel = await geoeffnet();
    panel.querySelector<HTMLInputElement>('#key-openai')!.value = 'sk-neu-9999';
    knopf(panel, 'openai', 'Speichern').click();
    await warte();
    expect(updateProvider).toHaveBeenCalledWith('zugangswort', 'openai',
      { api_key: 'sk-neu-9999' });
  });

  it('bietet Entfernen nur für hier hinterlegte Schlüssel', async () => {
    const panel = await geoeffnet();
    expect(knopf(panel, 'openai', 'Schlüssel entfernen')).toBeDefined();
    expect(knopf(panel, 'groq', 'Schlüssel entfernen')).toBeUndefined();
  });

  it('bietet Löschen nur für selbst eingetragene Anbieter', async () => {
    const panel = await geoeffnet();
    expect(knopf(panel, 'groq', 'Anbieter löschen')).toBeDefined();
    expect(knopf(panel, 'openai', 'Anbieter löschen')).toBeUndefined();
  });

  it('meldet das Ergebnis der Prüfung', async () => {
    testProvider.mockResolvedValue({ ok: true, detail: 'bereit.', latency_ms: 310 });
    const panel = await geoeffnet();
    knopf(panel, 'openai', 'Prüfen').click();
    await warte();
    expect(testProvider).toHaveBeenCalledWith('zugangswort', 'openai');
    expect(panel.querySelector('#einstellungen-meldung')!.textContent).toContain('310 ms');
  });

  it('meldet ein falsches Zugangswort und merkt es sich nicht', async () => {
    adminSettings.mockRejectedValue(new Error('Zugangswort stimmt nicht.'));
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'falsch';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();
    expect(panel.querySelector('#einstellungen-meldung')!.textContent)
      .toContain('Zugangswort stimmt nicht');
    expect(sessionStorage.getItem('xpertentisch.admin')).toBeNull();
  });
});

describe('Eigener Anbieter', () => {
  it('füllt die Felder aus einer Vorlage', async () => {
    const panel = await geoeffnet();
    panel.querySelector<HTMLButtonElement>('.vorschlaege button')!.click();
    expect(panel.querySelector<HTMLInputElement>('#neu-base')!.value)
      .toBe('https://api.groq.com/openai/v1');
    expect(panel.querySelector<HTMLInputElement>('#neu-model')!.value)
      .toBe('llama-3.3-70b-versatile');
  });

  it('legt einen Anbieter an', async () => {
    createProvider.mockResolvedValue(ANTWORT);
    const panel = await geoeffnet();
    panel.querySelector<HTMLInputElement>('#neu-label')!.value = 'Mein Server';
    panel.querySelector<HTMLInputElement>('#neu-base')!.value = 'http://127.0.0.1:11434/v1';
    panel.querySelector<HTMLInputElement>('#neu-model')!.value = 'llama3.1';
    panel.querySelector<HTMLInputElement>('#neu-key')!.value = 'egal';
    panel.querySelector<HTMLButtonElement>('#neu-anlegen')!.click();
    await warte();
    expect(createProvider).toHaveBeenCalledWith('zugangswort', {
      label: 'Mein Server', kind: 'openai', base_url: 'http://127.0.0.1:11434/v1',
      model: 'llama3.1', api_key: 'egal',
    });
  });

  it('verlangt Name und Modell', async () => {
    const panel = await geoeffnet();
    panel.querySelector<HTMLButtonElement>('#neu-anlegen')!.click();
    await warte();
    expect(createProvider).not.toHaveBeenCalled();
    expect(panel.querySelector('#einstellungen-meldung')!.textContent)
      .toContain('Anzeigename und Modellname');
  });

  it('bietet alle Arten von Schnittstellen an', async () => {
    const panel = await geoeffnet();
    const arten = [...panel.querySelectorAll<HTMLOptionElement>('#neu-kind option')]
      .map((o) => o.value);
    expect(arten).toEqual(['openai', 'anthropic', 'google']);
  });
});

describe('Fester Tisch im Entwicklungsbetrieb', () => {
  it('sperrt das Ändern und sagt warum', async () => {
    const panel = await geoeffnet({ ...ANTWORT, editable: false });
    expect(panel.textContent).toContain('steht der Tisch fest im Quelltext');
    expect(panel.querySelector('#neuer-anbieter')).toBeNull();
    expect(panel.querySelector<HTMLInputElement>('#an-openai')!.disabled).toBe(true);
  });
});
