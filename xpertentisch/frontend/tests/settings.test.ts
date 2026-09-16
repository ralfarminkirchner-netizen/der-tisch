import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { AdminSettings } from '../src/types';

const adminSettings = vi.fn();
const saveAdminSettings = vi.fn();
const testProvider = vi.fn();

vi.mock('../src/api', () => ({
  api: { adminSettings, saveAdminSettings, testProvider },
}));

const { renderSettings } = await import('../src/settings');

const ANTWORT: AdminSettings = {
  providers: [
    {
      provider: 'openai', label: 'OpenAI GPT', model: 'gpt-4.1',
      model_field: 'openai_model', key_field: 'openai_api_key', needs_key: true,
      key_source: 'einstellungen', key_hint: '…cdef', ready: true, reason: '',
    },
    {
      provider: 'anthropic', label: 'Anthropic Claude', model: 'claude-sonnet-4-5',
      model_field: 'anthropic_model', key_field: 'anthropic_api_key', needs_key: true,
      key_source: 'fehlt', key_hint: '', ready: false,
      reason: 'Kein Anthropic-Schlüssel hinterlegt — in den Einstellungen eintragen.',
    },
  ],
  request_timeout_s: 120,
  timeout_source: 'umgebung',
  env: 'production',
  fake_providers_enabled: false,
};

function warte(): Promise<void> {
  return new Promise((fertig) => setTimeout(fertig, 0));
}

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  document.body.replaceChildren();
});

describe('Einstellungen ohne eingerichtetes Zugangswort', () => {
  it('erklärt den Weg statt ein Feld anzubieten', () => {
    const panel = renderSettings(false, () => {});
    expect(panel.querySelector('#admin-token')).toBeNull();
    expect(panel.textContent).toContain('XT_ADMIN_TOKEN');
    expect(panel.querySelectorAll('.steps li')).toHaveLength(3);
  });

  it('nennt den Grund für den Schutz', () => {
    const panel = renderSettings(false, () => {});
    expect(panel.textContent).toContain('jede Person mit dem Link');
  });
});

describe('Einstellungen mit Zugangswort', () => {
  it('fragt zuerst das Zugangswort ab', () => {
    const panel = renderSettings(true, () => {});
    const feld = panel.querySelector<HTMLInputElement>('#admin-token');
    expect(feld).not.toBeNull();
    expect(feld!.type).toBe('password');
    expect(adminSettings).not.toHaveBeenCalled();
  });

  it('lädt den Zustand und zeigt beide Provider', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    const panel = renderSettings(true, () => {});
    document.body.append(panel);
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    expect(adminSettings).toHaveBeenCalledWith('zugangswort');
    const zeilen = panel.querySelectorAll('.settings-zeile[data-provider]');
    expect(zeilen).toHaveLength(2);
    expect(panel.textContent).toContain('OpenAI GPT');
    expect(panel.textContent).toContain('Anthropic Claude');
  });

  it('zeigt nur die Erkennungshilfe, nie einen Schlüssel', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    expect(panel.textContent).toContain('…cdef');
    const schluesselFeld = panel.querySelector<HTMLInputElement>('#key-openai')!;
    expect(schluesselFeld.type).toBe('password');
    expect(schluesselFeld.value).toBe('');
  });

  it('meldet den Grund, wenn ein Provider nicht bereit ist', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    const zeile = panel.querySelector('[data-provider="anthropic"]')!;
    expect(zeile.textContent).toContain('nicht bereit');
    expect(zeile.textContent).toContain('Kein Anthropic-Schlüssel hinterlegt');
  });

  it('bietet das Entfernen nur für hier hinterlegte Schlüssel an', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    const knopf = (zeile: string) =>
      [...panel.querySelectorAll<HTMLButtonElement>(`[data-provider="${zeile}"] button`)]
        .find((b) => b.textContent === 'Schlüssel entfernen')!;
    expect(knopf('openai').hasAttribute('disabled')).toBe(false);
    expect(knopf('anthropic').hasAttribute('disabled')).toBe(true);
  });

  it('schickt einen neuen Schlüssel und leert das Feld danach', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    saveAdminSettings.mockResolvedValue(ANTWORT);
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    const feld = panel.querySelector<HTMLInputElement>('#key-anthropic')!;
    feld.value = 'sk-ant-neu-1234';
    [...panel.querySelectorAll<HTMLButtonElement>('[data-provider="anthropic"] button')]
      .find((b) => b.textContent === 'Speichern')!.click();
    await warte();

    expect(saveAdminSettings).toHaveBeenCalledWith('zugangswort',
      { anthropic_api_key: 'sk-ant-neu-1234' });
    expect(panel.querySelector<HTMLInputElement>('#key-anthropic')!.value).toBe('');
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

  it('meldet das Ergebnis der Prüfung', async () => {
    adminSettings.mockResolvedValue(ANTWORT);
    testProvider.mockResolvedValue({ ok: true, detail: 'bereit.', latency_ms: 310 });
    const panel = renderSettings(true, () => {});
    panel.querySelector<HTMLInputElement>('#admin-token')!.value = 'zugangswort';
    panel.querySelector<HTMLButtonElement>('button.primary')!.click();
    await warte();

    [...panel.querySelectorAll<HTMLButtonElement>('[data-provider="openai"] button')]
      .find((b) => b.textContent?.startsWith('Prüfen'))!.click();
    await warte();

    expect(testProvider).toHaveBeenCalledWith('zugangswort', 'openai');
    expect(panel.querySelector('#einstellungen-meldung')!.textContent).toContain('310 ms');
  });
});
