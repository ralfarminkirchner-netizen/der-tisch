#!/usr/bin/env node
/**
 * Prüft den Produktionsbuild auf verräterische Inhalte.
 *
 * 1. Es dürfen keine (auch keine künstlichen) Zugangsdaten im Bundle stehen.
 * 2. Der Test-Provider darf nicht in den Auslieferungsstand geraten.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const DIST = resolve(process.argv[2] ?? 'dist');

const VERBOTEN = [
  { name: 'OpenAI-Schlüssel', re: /\bsk-[A-Za-z0-9_-]{16,}/ },
  { name: 'Anthropic-Schlüssel', re: /\bsk-ant-[A-Za-z0-9_-]{16,}/ },
  { name: 'Test-Secret', re: /(test|fake|dummy)[-_]?(secret|key|token|api[-_]?key)\s*[:=]\s*['"][^'"]{6,}/i },
  { name: 'Zugangsdaten-Variable', re: /\b(OPENAI_API_KEY|ANTHROPIC_API_KEY)\b/ },
  { name: 'Bearer-Token', re: /Bearer\s+[A-Za-z0-9._-]{20,}/ },
];

function* dateien(pfad) {
  for (const eintrag of readdirSync(pfad)) {
    const voll = join(pfad, eintrag);
    if (statSync(voll).isDirectory()) yield* dateien(voll);
    else yield voll;
  }
}

export function pruefeBuild(dist = DIST) {
  const funde = [];
  for (const datei of dateien(dist)) {
    if (!/\.(js|mjs|css|html|json|map)$/.test(datei)) continue;
    const inhalt = readFileSync(datei, 'utf8');
    for (const regel of VERBOTEN) {
      const treffer = inhalt.match(regel.re);
      if (treffer) funde.push(`${datei}: ${regel.name} (${treffer[0].slice(0, 24)}…)`);
    }
  }
  return funde;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const funde = pruefeBuild();
  if (funde.length > 0) {
    console.error('Build-Prüfung fehlgeschlagen:');
    for (const fund of funde) console.error(`  - ${fund}`);
    process.exit(1);
  }
  console.log('Build-Prüfung bestanden: keine Zugangsdaten im Auslieferungsstand.');
}
