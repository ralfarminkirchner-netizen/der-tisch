import { describe, expect, it } from 'vitest';

import { escapeHtml, highlightAnswer, markerIsValid, stripMarkup } from '../src/markers';
import type { Marker } from '../src/types';

function marker(over: Partial<Marker> & Pick<Marker, 'start_offset' | 'end_offset' | 'quote'>): Marker {
  return {
    id: 'm1',
    session_id: 's',
    spark_id: 'f',
    job_id: 'j1',
    related_job_id: null,
    kind: 'uebereinstimmung',
    note: 'Themenbezug',
    ...over,
  } as Marker;
}

const TEXT = 'Erster Satz mit Aussage. Zweiter Satz mit anderer Aussage.';

describe('highlightAnswer', () => {
  it('lässt den Originaltext unverändert', () => {
    const html = highlightAnswer(TEXT, [marker({ start_offset: 0, end_offset: 24, quote: TEXT.slice(0, 24) })]);
    expect(stripMarkup(html)).toBe(TEXT);
  });

  it('maskiert Markup aus der Antwort', () => {
    const boese = 'Antwort mit <script>alert("x")</script> darin.';
    const html = highlightAnswer(boese, []);
    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;');
    expect(stripMarkup(html)).toBe(boese);
  });

  it('maskiert auch innerhalb eines Markers', () => {
    const boese = '<img src=x onerror=alert(1)> und weiter.';
    const html = highlightAnswer(boese, [
      marker({ start_offset: 0, end_offset: boese.length, quote: boese, kind: 'einzigartig' }),
    ]);
    expect(html).not.toMatch(/<img/i);
    expect(html).toContain('<mark class="einzigartig"');
    expect(stripMarkup(html)).toBe(boese);
  });

  it('übergeht Marker mit falschem Textbeleg', () => {
    const html = highlightAnswer(TEXT, [
      marker({ start_offset: 0, end_offset: 10, quote: 'stimmt nicht' }),
    ]);
    expect(html).toBe(escapeHtml(TEXT));
    expect(html).not.toContain('<mark');
  });

  it('übergeht Marker außerhalb des Textes', () => {
    const html = highlightAnswer(TEXT, [
      marker({ start_offset: 500, end_offset: 900, quote: 'x' }),
    ]);
    expect(html).not.toContain('<mark');
  });

  it('zeigt bei Überschneidung den Widerspruch', () => {
    const html = highlightAnswer(TEXT, [
      marker({ id: 'a', start_offset: 0, end_offset: 24, quote: TEXT.slice(0, 24) }),
      marker({ id: 'b', start_offset: 0, end_offset: 24, quote: TEXT.slice(0, 24), kind: 'widerspruch' }),
    ]);
    expect(html).toContain('class="widerspruch"');
    expect(stripMarkup(html)).toBe(TEXT);
  });

  it('erhält Sonderzeichen und Zeilenumbrüche', () => {
    const text = 'Zeile eins & "zwei"\n\n  eingerückt < > 100 %';
    const html = highlightAnswer(text, []);
    expect(stripMarkup(html)).toBe(text);
  });
});

describe('markerIsValid', () => {
  it('prüft den Textbeleg', () => {
    expect(markerIsValid(TEXT, marker({ start_offset: 0, end_offset: 6, quote: 'Erster' }))).toBe(true);
    expect(markerIsValid(TEXT, marker({ start_offset: 0, end_offset: 6, quote: 'Falsch' }))).toBe(false);
  });
});
