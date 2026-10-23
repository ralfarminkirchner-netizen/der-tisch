import type { Summary, SummaryModelRow, SummaryPair } from './types';

const SVG_NS = 'http://www.w3.org/2000/svg';

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
 */
export function renderGraph(
  summary: Summary,
  onSelect: (selection: GraphSelection) => void,
): SVGSVGElement {
  const nodes = summary.models.filter((m) => summary.analysed_jobs.includes(m.job_id));
  const width = 360;
  const height = 280;
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
    text.setAttribute('fill', 'currentColor');
    text.textContent = 'Noch keine auswertbaren Antworten.';
    svg.appendChild(text);
    return svg;
  }

  const positions = layout(nodes, width, height);

  for (const pair of summary.pairs) {
    const a = positions.get(pair.a_job_id);
    const b = positions.get(pair.b_job_id);
    if (!a || !b) continue;
    if (pair.agreements > 0) {
      svg.appendChild(edge(pair, a, b, 'agree', pair.agreements, onSelect, -14));
    }
    if (pair.contradictions > 0) {
      svg.appendChild(edge(pair, a, b, 'contra', pair.contradictions, onSelect, 14));
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
  const cy = height / 2 - 8;
  const radius = Math.min(width, height) / 2 - 62;
  if (nodes.length === 1) {
    positions.set(nodes[0].job_id, { x: cx, y: cy });
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

function edge(
  pair: SummaryPair,
  a: { x: number; y: number },
  b: { x: number; y: number },
  kind: 'agree' | 'contra',
  count: number,
  onSelect: (selection: GraphSelection) => void,
  offset: number,
): SVGGElement {
  const group = document.createElementNS(SVG_NS, 'g');
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  const ox = (-dy / length) * offset;
  const oy = (dx / length) * offset;
  const x1 = a.x + ox;
  const y1 = a.y + oy;
  const x2 = b.x + ox;
  const y2 = b.y + oy;
  const mx = (x1 + x2) / 2 + ox * 0.18;
  const my = (y1 + y2) / 2 + oy * 0.18;

  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`);
  path.setAttribute('stroke-width', String(Math.min(2 + count, 7)));
  path.setAttribute('stroke-opacity', '0.88');
  path.setAttribute('fill', 'none');
  group.appendChild(path);

  // Eigene Trefferfläche: eine dünne Linie ist auf dem Telefon nicht treffbar
  // und für die Bedienung per Zeiger zu klein. Laufen zwei Kanten zwischen
  // denselben Knoten, rücken die Flächen zusätzlich längs auseinander —
  // sonst verdeckte die eine die andere und wäre nicht mehr anzutippen.
  const laengs = offset / 18;
  const hit = document.createElementNS(SVG_NS, 'circle');
  hit.setAttribute('cx', String((a.x + b.x) / 2 + ox + dx * laengs));
  hit.setAttribute('cy', String((a.y + b.y) / 2 + oy + dy * laengs));
  hit.setAttribute('r', '16');
  hit.setAttribute('class', 'edge-hit');
  group.appendChild(hit);

  const countText = document.createElementNS(SVG_NS, 'text');
  countText.setAttribute('x', String((a.x + b.x) / 2 + ox + dx * laengs));
  countText.setAttribute('y', String((a.y + b.y) / 2 + oy + dy * laengs + 4));
  countText.setAttribute('class', 'edge-count');
  countText.setAttribute('text-anchor', 'middle');
  countText.textContent = String(count);
  group.appendChild(countText);

  group.setAttribute('class', `edge ${kind}`);
  group.setAttribute('tabindex', '0');
  group.setAttribute('role', 'button');
  group.dataset.jobA = pair.a_job_id;
  group.dataset.jobB = pair.b_job_id;
  group.dataset.kind = kind;

  const label =
    kind === 'agree'
      ? `${count} Übereinstimmung(en): ${pair.a_label} ↔ ${pair.b_label}`
      : `${count} Widerspruch/Widersprüche: ${pair.a_label} ↔ ${pair.b_label}`;
  const title = document.createElementNS(SVG_NS, 'title');
  title.textContent = pair.topics.length ? `${label} — Themen: ${pair.topics.join('; ')}` : label;
  group.appendChild(title);
  group.setAttribute('aria-label', label);

  const select = () => onSelect({ jobIds: [pair.a_job_id, pair.b_job_id], label });
  group.addEventListener('click', select);
  group.addEventListener('keydown', (event) => {
    const key = (event as KeyboardEvent).key;
    if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      select();
    }
  });
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

  const radius = 16 + Math.min(node.sentences, 8);

  const hit = document.createElementNS(SVG_NS, 'circle');
  hit.setAttribute('cx', String(pos.x));
  hit.setAttribute('cy', String(pos.y));
  hit.setAttribute('r', '28');
  hit.setAttribute('fill', 'transparent');
  hit.setAttribute('class', 'node-hit');
  group.appendChild(hit);

  const ring = document.createElementNS(SVG_NS, 'circle');
  ring.setAttribute('cx', String(pos.x));
  ring.setAttribute('cy', String(pos.y));
  ring.setAttribute('r', String(radius + 6));
  ring.setAttribute('class', 'fokusring');
  group.appendChild(ring);

  const circle = document.createElementNS(SVG_NS, 'circle');
  circle.setAttribute('cx', String(pos.x));
  circle.setAttribute('cy', String(pos.y));
  circle.setAttribute('r', String(radius));
  circle.setAttribute('class', 'knoten');
  group.appendChild(circle);

  const text = document.createElementNS(SVG_NS, 'text');
  text.setAttribute('x', String(pos.x));
  text.setAttribute('y', String(pos.y + radius + 14));
  text.textContent = shorten(node.label);
  group.appendChild(text);

  const title = document.createElementNS(SVG_NS, 'title');
  title.textContent =
    `${node.label}: ${node.sentences} Sätze, ${node.agreements} Übereinstimmungen, ` +
    `${node.contradictions} Widersprüche, ${node.unique} einzigartig`;
  group.appendChild(title);

  const select = () => onSelect({ jobIds: [node.job_id], label: node.label });
  group.addEventListener('click', select);
  group.addEventListener('keydown', (event) => {
    const key = (event as KeyboardEvent).key;
    if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      select();
    }
  });
  return group;
}

function shorten(label: string): string {
  return label.length > 14 ? `${label.slice(0, 13)}…` : label;
}
