import { existsSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

// @ts-expect-error – reines JS-Hilfsskript ohne Typdeklaration
import { pruefeBuild } from '../scripts/check-build.mjs';

const DIST = resolve(__dirname, '../dist');

describe('Produktionsbuild', () => {
  it('erkennt künstliche Test-Secrets zuverlässig', () => {
    const dir = mkdtempSync(join(tmpdir(), 'xt-build-'));
    writeFileSync(join(dir, 'app.js'), 'const TEST_SECRET = "abc123456789";');
    writeFileSync(join(dir, 'other.js'), 'const k = "sk-ant-0123456789abcdefghij";');
    const funde = pruefeBuild(dir) as string[];
    // Beide Dateien müssen beanstandet werden (ein Fund kann mehrere Regeln treffen).
    expect(funde.some((f) => f.includes('app.js'))).toBe(true);
    expect(funde.some((f) => f.includes('other.js'))).toBe(true);
  });

  it('enthält keine Zugangsdaten', () => {
    expect(
      existsSync(DIST),
      'dist/ fehlt — bitte zuerst "npm run build" ausführen',
    ).toBe(true);
    expect(pruefeBuild(DIST)).toEqual([]);
  });

  it('bringt den Test-Provider nicht in den Auslieferungsstand', () => {
    expect(existsSync(DIST)).toBe(true);
    const sammeln = (p: string): string[] =>
      readdirSync(p).flatMap((e: string) => {
        const voll = join(p, e);
        return statSync(voll).isDirectory() ? sammeln(voll) : [voll];
      });
    const inhalt = sammeln(DIST)
      .filter((f) => /\.(js|mjs|html)$/.test(f))
      .map((f) => readFileSync(f, 'utf8'))
      .join('\n');
    expect(inhalt).not.toContain('FakeProvider');
    expect(inhalt).not.toContain('XT_ALLOW_FAKE_PROVIDERS');
  });
});
