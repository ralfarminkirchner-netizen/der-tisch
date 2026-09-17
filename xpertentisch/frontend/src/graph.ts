import type { Summary, SummaryModelRow, SummaryPair } from './types';

export const SVG_NS = 'http://www.w3.org/2000/svg';

/** Ein Knoten braucht Platz für seinen Kreis UND seine Beschriftung darunter. */
const RAND = 40;
const KNOTEN_MIN = 17;
/** Die Trefferfläche einer Kante. Groß genug für einen Daumen. */
const TREFFER = 17;
/** Wie weit sich die beiden Kanten eines Paares voneinander wegbiegen. */
const BOGEN = 46;

export interface GraphSelection {
  /** Aufträge, die durch den Klick geöffnet werden sollen. */
  jobIds: string[];
  label: string;
}

/**
 * Zeichnet das Beziehungsnetz der Antworten.
 *
 * Knoten sind die Modelle eines Funkens, Kanten die gemeinsamen bzw.
 * gegensätzlichen Aussagen. Ein Klick liefert genau die Aufträge, die zu dem
 * angeklickten Element gehören — bei einem Knoten dessen eigene Antwort, bei
 * einer Kante beide beteiligten Antworten.
 *
 * Gestalterisch: die Knoten stehen als Ring, damit keiner oben und keiner
 * unten steht — es gibt hier keine Rangfolge. Themenbezug-Hinweis und Gegensatzhinweis
 * zwischen denselben zwei Stimmen biegen sich voneinander weg, damit beide
 * getrennt sichtbar und getrennt treffbar bleiben.
 */
export function renderGraph(
  summary: Summary,
  onSelect: (selection: GraphSelection) => void,
): SVGSVGElement {
  const nodes = summary.models.filter((m) => summary.analysed_jobs.includes(m.job_id));
  const width = 380;
  const height = 300;
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Beziehungsnetz der Antworten');

  if (nodes.length === 0) {
    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', String(width / 2));
    text.setAttribute('y', String(height / 2));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('class', 'leer');
    text.textContent = 'Noch keine auswertbaren Antworten.';
    svg.appendChild(text);
    return svg;
  }

  const positions = layout(nodes, width, height);

  // Kanten zuerst: Knoten liegen darüber und bleiben lesbar.
  for (const pair of summary.pairs) {
    const a = positions.get(pair.a_job_id);
    const b = positions.get(pair.b_job_id);
    if (!a || !b) continue;
    if (pair.agreements > 0) {
      svg.appendChild(edge(pair, a, b, 'agree', pair.agreements, onSelect, -BOGEN));
    }
    if (pair.contradictions > 0) {
      svg.appendChild(edge(pair, a, b, 'contra', pair.contradictions, onSelect, BOGEN));
    }
  }

  for (const node of nodes) {
    const pos = positions.get(node.job_id);
    if (!pos) continue;
    svg.appendChild(nodeElement(node, pos, onSelect));
  }

  return svg;
}

function layout(
  nodes: SummaryModelRow[],
  width: number,
  height: number,
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) / 2 - RAND;
  if (nodes.length === 1) {
    positions.set(nodes[0].job_id, { x: cx, y: cy });
    return positions;
  }
  if (nodes.length === 2) {
    // Zwei Stimmen stehen sich gegenüber; ein Ring wäre hier nur Behauptung.
    positions.set(nodes[0].job_id, { x: cx - radius, y: cy });
    positions.set(nodes[1].job_id, { x: cx + radius, y: cy });
    return positions;
  }
  nodes.forEach((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2 - Math.PI / 2;
    positions.set(node.job_id, {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
    });
  });
  return positions;
}

/** Der Punkt in der Mitte einer quadratischen Bézierkurve (t = 0,5). */
function bogenmitte(
  a: { x: number; y: number },
  s: { x: number; y: number },
  b: { x: number; y: number },
): { x: number; y: number } {
  return { x: (a.x + 2 * s.x + b.x) / 4, y: (a.y + 2 * s.y + b.y) / 4 };
}

function edge(
  pair: SummaryPair,
  a: { x: number; y: number },
  b: { x: number; y: number },
  kind: 'agree' | 'contra',
  count: number,
  onSelect: (selection: GraphSelection) => void,
  bogen: number,
): SVGGElement {
  const group = document.createElementNS(SVG_NS, 'g');
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  // Steuerpunkt senkrecht zur Verbindung. Die beiden Arten biegen in
  // entgegengesetzte Richtungen und können sich darum nicht verdecken.
  const steuer = {
    x: (a.x + b.x) / 2 + (-dy / length) * bogen,
    y: (a.y + b.y) / 2 + (dx / length) * bogen,
  };

  const pfad = document.createElementNS(SVG_NS, 'path');
  pfad.setAttribute('d', `M ${a.x} ${a.y} Q ${steuer.x} ${steuer.y} ${b.x} ${b.y}`);
  pfad.setAttribute('stroke-width', String(Math.min(2 + count, 6)));
  pfad.setAttribute('stroke-opacity', '0.9');
  group.appendChild(pfad);

  // Eigene Trefferfläche an der Bogenmitte: eine dünne Linie ist auf dem
  // Telefon nicht treffbar. Weil die Bögen auseinanderlaufen, liegen auch die
  // Trefferflächen zweier Kanten desselben Paares weit genug auseinander.
  const mitte = bogenmitte(a, steuer, b);
  const hit = document.createElementNS(SVG_NS, 'circle');
  hit.setAttribute('cx', String(mitte.x));
  hit.setAttribute('cy', String(mitte.y));
  hit.setAttribute('r', String(TREFFER));
  hit.setAttribute('class', 'edge-hit');
  group.appendChild(hit);

  // Die Anzahl steht als Marke an der Kante: wie oft diese beiden Stimmen
  // sich hier treffen bzw. widersprechen. Die Scheibe darunter macht sie auch
  // dort lesbar, wo eine andere Kante hindurchläuft.
  const scheibe = document.createElementNS(SVG_NS, 'circle');
  scheibe.setAttribute('cx', String(mitte.x));
  scheibe.setAttribute('cy', String(mitte.y));
  scheibe.setAttribute('r', '10');
  scheibe.setAttribute('class', 'edge-scheibe');
  group.appendChild(scheibe);

  const zahl = document.createElementNS(SVG_NS, 'text');
  zahl.setAttribute('x', String(mitte.x));
  zahl.setAttribute('y', String(mitte.y + 3.5));
  zahl.setAttribute('class', 'edge-zahl');
  zahl.textContent = String(count);
  group.appendChild(zahl);

  group.setAttribute('class', `edge ${kind}`);
  group.setAttribute('tabindex', '0');
  group.setAttribute('role', 'button');
  group.dataset.jobA = pair.a_job_id;
  group.dataset.jobB = pair.b_job_id;
  group.dataset.kind = kind;

  const label =
    kind === 'agree'
      ? `${count} Themenbezug-Hinweis(e): ${pair.a_label} ↔ ${pair.b_label}`
      : `${count} Gegensatzhinweis(e): ${pair.a_label} ↔ ${pair.b_label}`;
  const title = document.createElementNS(SVG_NS, 'title');
  title.textContent = pair.topics.length ? `${label} — Themen: ${pair.topics.join('; ')}` : label;
  group.appendChild(title);
  group.setAttribute('aria-label', label);

  bedienbar(group, () => onSelect({ jobIds: [pair.a_job_id, pair.b_job_id], label }));
  return group;
}

function nodeElement(
  node: SummaryModelRow,
  pos: { x: number; y: number },
  onSelect: (selection: GraphSelection) => void,
): SVGGElement {
  const group = document.createElementNS(SVG_NS, 'g');
  group.setAttribute('class', 'node');
  group.setAttribute('tabindex', '0');
  group.setAttribute('role', 'button');
  group.setAttribute('aria-label', `Antwort von ${node.label} öffnen`);
  group.dataset.jobId = node.job_id;

  // Die Größe zeigt den Umfang der Antwort — sie ist keine Bewertung.
  const r = KNOTEN_MIN + Math.min(node.sentences, 10);
  const circle = document.createElementNS(SVG_NS, 'circle');
  circle.setAttribute('cx', String(pos.x));
  circle.setAttribute('cy', String(pos.y));
  circle.setAttribute('r', String(r));
  group.appendChild(circle);

  // Im Kreis das Signet, darunter der Name — dieselbe Ordnung wie auf der Karte.
  const signet = document.createElementNS(SVG_NS, 'text');
  signet.setAttribute('x', String(pos.x));
  signet.setAttribute('y', String(pos.y + 4));
  signet.textContent = initialen(node.label);
  group.appendChild(signet);

  const name = document.createElementNS(SVG_NS, 'text');
  name.setAttribute('x', String(pos.x));
  name.setAttribute('y', String(pos.y + r + 14));
  name.setAttribute('class', 'zahl');
  name.textContent = shorten(node.label);
  group.appendChild(name);

  const title = document.createElementNS(SVG_NS, 'title');
  title.textContent =
    `${node.label}: ${node.sentences} Sätze, ${node.agreements} Themenbezüge (Hinweis), ` +
    `${node.contradictions} Gegensatzhinweise, ${node.unique} ohne Treffer in diesem Verfahren`;
  group.appendChild(title);

  bedienbar(group, () => onSelect({ jobIds: [node.job_id], label: node.label }));
  return group;
}

/** Klick und Tastatur führen zur selben Handlung. */
export function bedienbar(group: SVGGElement, handeln: () => void): void {
  group.addEventListener('click', handeln);
  group.addEventListener('keydown', (event) => {
    const key = (event as KeyboardEvent).key;
    if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      handeln();
    }
  });
}

export function initialen(label: string): string {
  return (
    label
      .split(/[\s—–-]+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((teil) => teil[0])
      .join('')
      .toUpperCase() || '?'
  );
}

export function shorten(label: string): string {
  return label.length > 16 ? `${label.slice(0, 15)}…` : label;
}
