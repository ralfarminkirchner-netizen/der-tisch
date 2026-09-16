import type { Marker, MarkerKind } from './types';

/** Reihenfolge bei Überschneidungen: der auffälligste Befund gewinnt. */
const PRIORITY: Record<MarkerKind, number> = {
  widerspruch: 3,
  uebereinstimmung: 2,
  einzigartig: 1,
};

export const KIND_LABEL: Record<MarkerKind, string> = {
  widerspruch: 'Widerspruch',
  uebereinstimmung: 'Übereinstimmung',
  einzigartig: 'Einzigartig',
};

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

interface Cover {
  kind: MarkerKind;
  note: string;
  markerId: string;
  relatedJobId: string | null;
}

/** Prüft, ob ein Marker sauber auf den Text zeigt und sein Zitat wirklich dort steht. */
export function markerIsValid(text: string, marker: Marker): boolean {
  const { start_offset: start, end_offset: end } = marker;
  if (!Number.isInteger(start) || !Number.isInteger(end)) return false;
  if (start < 0 || end > text.length || start >= end) return false;
  return text.slice(start, end) === marker.quote;
}

/**
 * Legt die Marker als Auszeichnung über den Text — ohne ihn zu verändern.
 *
 * Der Rückgabewert ist HTML: jeder Textteil ist maskiert, Marker werden als
 * <mark> umschlossen. Entfernt man die Auszeichnung wieder, ergibt sich exakt
 * der Originaltext. Marker mit ungültigem Bezug werden stillschweigend
 * übergangen, statt den Text zu verfälschen.
 */
export function highlightAnswer(text: string, markers: Marker[]): string {
  if (!text) return '';
  const usable = markers.filter((m) => markerIsValid(text, m));
  if (usable.length === 0) return escapeHtml(text);

  const cover: (Cover | null)[] = new Array(text.length).fill(null);
  const sorted = [...usable].sort((a, b) => PRIORITY[a.kind] - PRIORITY[b.kind]);
  for (const marker of sorted) {
    for (let i = marker.start_offset; i < marker.end_offset; i += 1) {
      const current = cover[i];
      if (current === null || PRIORITY[marker.kind] >= PRIORITY[current.kind]) {
        cover[i] = {
          kind: marker.kind,
          note: marker.note,
          markerId: marker.id,
          relatedJobId: marker.related_job_id,
        };
      }
    }
  }

  const parts: string[] = [];
  let index = 0;
  while (index < text.length) {
    const current = cover[index];
    let end = index + 1;
    while (end < text.length && sameCover(cover[end], current)) end += 1;
    const segment = escapeHtml(text.slice(index, end));
    if (current === null) {
      parts.push(segment);
    } else {
      const title = `${KIND_LABEL[current.kind]} — ${current.note}`;
      parts.push(
        `<mark class="${current.kind}" data-marker="${escapeHtml(current.markerId)}"` +
          (current.relatedJobId
            ? ` data-related="${escapeHtml(current.relatedJobId)}"`
            : '') +
          ` title="${escapeHtml(title)}">${segment}</mark>`,
      );
    }
    index = end;
  }
  return parts.join('');
}

function sameCover(a: Cover | null, b: Cover | null): boolean {
  if (a === null || b === null) return a === b;
  return a.markerId === b.markerId;
}

/** Entfernt die Auszeichnung wieder — Grundlage der Unverändertheits-Prüfung. */
export function stripMarkup(html: string): string {
  return html
    .replace(/<[^>]*>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}
