/**
 * Linsen auf dieselben Daten.
 *
 * Das Beziehungsnetz zeigte bisher genau eine Sache: welche Stimme welcher
 * widerspricht. Dieselben Daten tragen mehr — worüber gestritten wird, was nur
 * einer sagt, und welche Wege im Gespräch tatsächlich verfolgt wurden.
 *
 * Jede Linse ist eine andere Projektion **vorhandener** Daten. Keine Linse
 * rechnet etwas hinzu, schätzt oder sagt etwas voraus: was man sieht, steht so
 * in den Antworten, den Fundstellen oder den gesetzten Bezügen.
 */

import { SVG_NS, bedienbar, initialen, shorten } from './graph';
import type { GraphSelection } from './graph';
import type {
  Folge,
  Job,
  Marker,
  Relation,
  SessionBundle,
  SparkEntry,
  Summary,
  Szenario,
} from './types';

export type LinsenArt = 'stimmen' | 'themen' | 'szenario' | 'herkunft' | 'zeit';

export const LINSEN: { id: LinsenArt; name: string; erklaerung: string }[] = [
  {
    id: 'stimmen',
    name: 'Stimmen',
    erklaerung:
      'Wer trifft sich mit wem, und wo widersprechen sie einander? Jede Kante ' +
      'steht für ausgewiesene Fundstellen in beiden Antworten.',
  },
  {
    id: 'themen',
    name: 'Themen',
    erklaerung:
      'Woran hängen die Befunde? Links die Stimmen, rechts die Begriffe, an ' +
      'denen sie sich treffen, streiten oder alleine stehen.',
  },
  {
    id: 'szenario',
    name: 'Folgen',
    erklaerung:
      'Was folgt daraus, laut den Stimmen? Links die Ausgangsaussage, rechts die ' +
      'genannten Folgen. Die Zahl sagt, wie viele Stimmen eine Folge genannt haben — ' +
      'das ist eine Häufigkeit, keine Wahrscheinlichkeit.',
  },
  {
    id: 'herkunft',
    name: 'Herkunft',
    erklaerung:
      'Worauf kannst du dich berufen? Durchgezogen, was du bestätigt hast; ' +
      'gestrichelt, was die Auswertung bloß vorschlägt; ausgegraut, was du ' +
      'verworfen hast.',
  },
  {
    id: 'zeit',
    name: 'Zeit',
    erklaerung:
      'Wie lief diese Runde ab? Wer wann anfing, wie lange schrieb, wo die ' +
      'Warteschlange bremste. Die Zeilen stehen in der Reihenfolge des Tisches — ' +
      'schneller ist hier nicht besser.',
  },
];

/** Der Verlauf ist sitzungsweit und steht darum außerhalb der Linsenwahl. */
export const VERLAUF_ERKLAERUNG =
  'Welche Wege wurden verfolgt? Jeder Zweig ist ein Beitrag, der aus einem ' +
  'anderen hervorgegangen ist — keine Vorhersage, sondern das, was geschah.';

function text(
  x: number, y: number, inhalt: string, klasse = '', anker = 'middle',
): SVGTextElement {
  const t = document.createElementNS(SVG_NS, 'text');
  t.setAttribute('x', String(x));
  t.setAttribute('y', String(y));
  t.setAttribute('text-anchor', anker);
  if (klasse) t.setAttribute('class', klasse);
  t.textContent = inhalt;
  return t;
}

function leereFlaeche(svg: SVGSVGElement, breite: number, hoehe: number, satz: string): void {
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);
  svg.appendChild(text(breite / 2, hoehe / 2, satz, 'leer'));
}

// ------------------------------------------------------------- Themen-Linse

interface ThemenKnoten {
  begriff: string;
  /** Job-Kennungen, die an diesem Begriff hängen — je Art getrennt. */
  einig: Set<string>;
  gegen: Set<string>;
  einzeln: Set<string>;
}

/** Sammelt die Begriffe aus den Fundstellen. Nichts wird dazuerfunden. */
export function themenAusMarkern(markers: Marker[]): ThemenKnoten[] {
  const nach = new Map<string, ThemenKnoten>();
  for (const marker of markers) {
    for (const begriff of marker.topics ?? []) {
      if (!begriff) continue;
      let knoten = nach.get(begriff);
      if (!knoten) {
        knoten = { begriff, einig: new Set(), gegen: new Set(), einzeln: new Set() };
        nach.set(begriff, knoten);
      }
      if (marker.kind === 'uebereinstimmung') knoten.einig.add(marker.job_id);
      else if (marker.kind === 'widerspruch') knoten.gegen.add(marker.job_id);
      else knoten.einzeln.add(marker.job_id);
    }
  }
  // Schwere zuerst: woran die meisten Stimmen hängen, steht oben. Bei
  // Gleichstand entscheidet das Alphabet, damit die Reihenfolge stabil bleibt.
  return [...nach.values()].sort((a, b) => {
    const ga = a.einig.size + a.gegen.size + a.einzeln.size;
    const gb = b.einig.size + b.gegen.size + b.einzeln.size;
    return gb - ga || a.begriff.localeCompare(b.begriff, 'de');
  });
}

const THEMEN_HOECHSTENS = 9;

export function renderThemen(
  summary: Summary,
  markers: Marker[],
  onSelect: (selection: GraphSelection) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph themen');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Themen und Stimmen');

  const stimmen = summary.models.filter((m) => summary.analysed_jobs.includes(m.job_id));
  const alle = themenAusMarkern(markers);
  const themen = alle.slice(0, THEMEN_HOECHSTENS);

  if (stimmen.length === 0 || themen.length === 0) {
    leereFlaeche(svg, 380, 160, 'Noch keine Begriffe gefunden.');
    return svg;
  }

  const zeile = 34;
  const oben = 26;
  const hoehe = Math.max(stimmen.length, themen.length) * zeile + oben * 2;
  const breite = 380;
  const xStimme = 54;
  const xThema = 250;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  const yStimme = new Map<string, number>();
  stimmen.forEach((m, i) => {
    const mitte = (hoehe - stimmen.length * zeile) / 2;
    yStimme.set(m.job_id, mitte + i * zeile + zeile / 2);
  });
  const yThema = new Map<string, number>();
  themen.forEach((t, i) => {
    const mitte = (hoehe - themen.length * zeile) / 2;
    yThema.set(t.begriff, mitte + i * zeile + zeile / 2);
  });

  // Kanten zuerst, damit die Knoten darüber lesbar bleiben.
  for (const thema of themen) {
    const zy = yThema.get(thema.begriff)!;
    const arten: [Set<string>, string][] = [
      [thema.einig, 'agree'],
      [thema.gegen, 'contra'],
      [thema.einzeln, 'unique'],
    ];
    for (const [menge, art] of arten) {
      for (const jobId of menge) {
        const sy = yStimme.get(jobId);
        if (sy === undefined) continue;
        const pfad = document.createElementNS(SVG_NS, 'path');
        const mx = (xStimme + xThema) / 2;
        pfad.setAttribute('d', `M ${xStimme + 16} ${sy} C ${mx} ${sy}, ${mx} ${zy}, ${xThema - 8} ${zy}`);
        pfad.setAttribute('class', `faden ${art}`);
        svg.appendChild(pfad);
      }
    }
  }

  // Welche Stimmen überhaupt an einem der gezeigten Begriffe hängen.
  const angebunden = new Set<string>();
  for (const thema of themen) {
    for (const id of [...thema.einig, ...thema.gegen, ...thema.einzeln]) angebunden.add(id);
  }

  for (const stimme of stimmen) {
    const y = yStimme.get(stimme.job_id)!;
    const ohne = !angebunden.has(stimme.job_id);
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', `node stimme${ohne ? ' ohne-faden' : ''}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.setAttribute('aria-label', `Antwort von ${stimme.label} öffnen`);
    gruppe.dataset.jobId = stimme.job_id;

    const kreis = document.createElementNS(SVG_NS, 'circle');
    kreis.setAttribute('cx', String(xStimme));
    kreis.setAttribute('cy', String(y));
    kreis.setAttribute('r', '14');
    gruppe.appendChild(kreis);
    gruppe.appendChild(text(xStimme, y + 4, initialen(stimme.label)));
    gruppe.appendChild(text(xStimme - 22, y + 4, shorten(stimme.label), 'zahl', 'end'));

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent = ohne
      ? `${stimme.label}: berührt keinen der gezeigten Begriffe.`
      : `${stimme.label}: ${stimme.agreements} Übereinstimmungen, ` +
        `${stimme.contradictions} Widersprüche, ${stimme.unique} einzigartig`;
    gruppe.appendChild(titel);

    bedienbar(gruppe, () => onSelect({ jobIds: [stimme.job_id], label: stimme.label }));
    svg.appendChild(gruppe);
  }

  for (const thema of themen) {
    const y = yThema.get(thema.begriff)!;
    const beteiligt = new Set([...thema.einig, ...thema.gegen, ...thema.einzeln]);
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', 'node thema');
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    const wie = thema.gegen.size > 0
      ? 'umstritten'
      : thema.einig.size > 0 ? 'geteilt' : 'einzeln genannt';
    gruppe.setAttribute(
      'aria-label',
      `Begriff „${thema.begriff}", ${wie}, ${beteiligt.size} Stimmen — Antworten öffnen`,
    );
    gruppe.dataset.thema = thema.begriff;

    // Ein Begriff, der umstritten ist, bekommt die Widerspruchsfarbe — das ist
    // dieselbe Bedeutung wie überall sonst, keine zweite Bedeutung der Farbe.
    const r = 4 + Math.min(beteiligt.size, 5);
    const punkt = document.createElementNS(SVG_NS, 'circle');
    punkt.setAttribute('cx', String(xThema));
    punkt.setAttribute('cy', String(y));
    punkt.setAttribute('r', String(r));
    punkt.setAttribute(
      'class',
      `themapunkt ${thema.gegen.size > 0 ? 'contra' : thema.einig.size > 0 ? 'agree' : 'unique'}`,
    );
    gruppe.appendChild(punkt);
    // Der Name beginnt hinter dem Punkt, nicht auf ihm — sonst überdeckte die
    // Größe des Punktes die ersten Buchstaben.
    gruppe.appendChild(text(xThema + r + 7, y + 4, thema.begriff, 'themaname', 'start'));

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent =
      `„${thema.begriff}" — ${thema.einig.size} einig, ${thema.gegen.size} im Widerspruch, ` +
      `${thema.einzeln.size} allein`;
    gruppe.appendChild(titel);

    bedienbar(gruppe, () =>
      onSelect({ jobIds: [...beteiligt], label: `Begriff „${thema.begriff}"` }),
    );
    svg.appendChild(gruppe);
  }

  if (alle.length > themen.length) {
    svg.appendChild(
      text(breite / 2, hoehe - 8, `+ ${alle.length - themen.length} weitere Begriffe`, 'leer'),
    );
  }

  return svg;
}

// ------------------------------------------------------------ Verlaufsbaum

interface Zweig {
  entry: SparkEntry;
  tiefe: number;
  elternId: string | null;
}

/**
 * Ordnet die Beiträge einer Sitzung zu einem Baum.
 *
 * Ein Beitrag hängt an dem Beitrag, aus dessen Antwort er hervorgegangen ist —
 * das steht in seinen ausdrücklich gesetzten Bezügen. Ohne Bezug steht er als
 * eigener Ursprung da.
 */
export function baueZweige(sparks: SparkEntry[]): Zweig[] {
  const sparkVonJob = new Map<string, string>();
  for (const eintrag of sparks) {
    for (const job of eintrag.jobs) sparkVonJob.set(job.id, eintrag.spark.id);
  }

  const elternVon = new Map<string, string | null>();
  for (const eintrag of sparks) {
    let eltern: string | null = null;
    for (const ref of eintrag.spark.refs ?? []) {
      const quelle = sparkVonJob.get(ref);
      if (quelle && quelle !== eintrag.spark.id) {
        eltern = quelle;
        break;
      }
    }
    elternVon.set(eintrag.spark.id, eltern);
  }

  const tiefeVon = new Map<string, number>();
  const tiefe = (id: string, gesehen = new Set<string>()): number => {
    if (tiefeVon.has(id)) return tiefeVon.get(id)!;
    if (gesehen.has(id)) return 0; // Schutz gegen Ringe in den Bezügen
    gesehen.add(id);
    const eltern = elternVon.get(id) ?? null;
    const wert = eltern ? tiefe(eltern, gesehen) + 1 : 0;
    tiefeVon.set(id, wert);
    return wert;
  };

  return sparks.map((eintrag) => ({
    entry: eintrag,
    tiefe: tiefe(eintrag.spark.id),
    elternId: elternVon.get(eintrag.spark.id) ?? null,
  }));
}

const ART_KURZ: Record<string, string> = {
  funke: 'Funke',
  antwort: 'Antwort',
  weitergabe: 'Weitergabe',
  gegenposition: 'Gegenposition',
  vertiefung: 'Vertiefung',
  szenario: 'Szenario',
  kuratierung: 'Kuratierung',
  pingpong: 'Wechselrede',
};

export function renderVerlauf(
  bundle: SessionBundle,
  aktuellerSpark: string | null,
  onSelect: (sparkId: string) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph verlauf');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Verlauf der Sitzung');

  const zweige = baueZweige(bundle.sparks);
  if (zweige.length === 0) {
    leereFlaeche(svg, 380, 120, 'Noch kein Beitrag gesetzt.');
    return svg;
  }

  const kastenBreite = 128;
  const kastenHoehe = 38;
  const spaltenAbstand = 46;
  const zeilenAbstand = 14;
  const rand = 12;

  const maxTiefe = Math.max(...zweige.map((z) => z.tiefe));
  const breite = rand * 2 + (maxTiefe + 1) * kastenBreite + maxTiefe * spaltenAbstand;
  const hoehe = rand * 2 + zweige.length * (kastenHoehe + zeilenAbstand) - zeilenAbstand;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);
  // Natürliche Größe mitgeben: in der schmalen Spalte des Linsenraums soll der
  // Baum scrollen, nicht schrumpfen — sonst wäre seine Beschriftung zu klein.
  svg.setAttribute('width', String(breite));
  svg.setAttribute('height', String(hoehe));

  const pos = new Map<string, { x: number; y: number }>();
  zweige.forEach((zweig, i) => {
    pos.set(zweig.entry.spark.id, {
      x: rand + zweig.tiefe * (kastenBreite + spaltenAbstand),
      y: rand + i * (kastenHoehe + zeilenAbstand),
    });
  });

  // Verbindungen: rechtwinklig, damit die Abzweigung als Abzweigung lesbar ist.
  for (const zweig of zweige) {
    if (!zweig.elternId) continue;
    const von = pos.get(zweig.elternId);
    const nach = pos.get(zweig.entry.spark.id);
    if (!von || !nach) continue;
    const x1 = von.x + kastenBreite;
    const y1 = von.y + kastenHoehe / 2;
    const x2 = nach.x;
    const y2 = nach.y + kastenHoehe / 2;
    const mx = x1 + spaltenAbstand / 2;
    const pfad = document.createElementNS(SVG_NS, 'path');
    pfad.setAttribute('d', `M ${x1} ${y1} H ${mx} V ${y2} H ${x2}`);
    pfad.setAttribute('class', 'ast');
    svg.appendChild(pfad);
  }

  for (const zweig of zweige) {
    const { spark, jobs } = zweig.entry;
    const p = pos.get(spark.id)!;
    const gruppe = document.createElementNS(SVG_NS, 'g');
    const aktuell = spark.id === aktuellerSpark;
    gruppe.setAttribute('class', `zweig${aktuell ? ' aktuell' : ''}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.dataset.sparkId = spark.id;

    const kasten = document.createElementNS(SVG_NS, 'rect');
    kasten.setAttribute('x', String(p.x));
    kasten.setAttribute('y', String(p.y));
    kasten.setAttribute('width', String(kastenBreite));
    kasten.setAttribute('height', String(kastenHoehe));
    kasten.setAttribute('rx', '8');
    gruppe.appendChild(kasten);

    gruppe.appendChild(
      text(p.x + 10, p.y + 15, ART_KURZ[spark.kind] ?? spark.kind, 'zweigart', 'start'),
    );
    gruppe.appendChild(
      text(p.x + 10, p.y + 29, gekuerzt(spark.prompt, 20), 'zweigtext', 'start'),
    );

    // Ein Strich je Antwort — der Zustand der Runde auf einen Blick, ohne Zahl.
    jobs.slice(0, 8).forEach((job, i) => {
      const strich = document.createElementNS(SVG_NS, 'rect');
      strich.setAttribute('x', String(p.x + kastenBreite - 9 - i * 5));
      strich.setAttribute('y', String(p.y + 8));
      strich.setAttribute('width', '3');
      strich.setAttribute('height', '8');
      strich.setAttribute('rx', '1.5');
      strich.setAttribute('class', `zustandsstrich zustand-${job.status}`);
      gruppe.appendChild(strich);
    });

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent =
      `${ART_KURZ[spark.kind] ?? spark.kind} ${spark.seq}: ${spark.prompt}` +
      `\n${jobs.length} Antwort(en)`;
    gruppe.appendChild(titel);
    gruppe.setAttribute(
      'aria-label',
      `${ART_KURZ[spark.kind] ?? spark.kind} ${spark.seq}, ${jobs.length} Antworten — hingehen`,
    );

    bedienbar(gruppe, () => onSelect(spark.id));
    svg.appendChild(gruppe);
  }

  return svg;
}

function gekuerzt(satz: string, zeichen: number): string {
  const sauber = satz.replace(/\s+/g, ' ').trim();
  return sauber.length > zeichen ? `${sauber.slice(0, zeichen - 1)}…` : sauber;
}

// ---------------------------------------------------------- Szenario-Linse

/**
 * Bricht einen Satz in Zeilen, die in die Fläche passen.
 *
 * Handgemacht, weil SVG von sich aus keinen Umbruch kennt. Ein zu langes Wort
 * wird nicht zerschnitten — lieber eine überstehende Zeile als ein zerrissenes
 * Wort; die Fläche ist dafür breit genug bemessen.
 */
export function umbruch(satz: string, zeichen: number, hoechstens: number): string[] {
  const woerter = satz.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  if (woerter.length === 0) return [];
  const zeilen: string[] = [];
  let laufend = '';
  for (const wort of woerter) {
    const versuch = laufend ? `${laufend} ${wort}` : wort;
    if (versuch.length <= zeichen || !laufend) {
      laufend = versuch;
    } else {
      zeilen.push(laufend);
      laufend = wort;
    }
  }
  zeilen.push(laufend);
  if (zeilen.length <= hoechstens) return zeilen;
  const gekappt = zeilen.slice(0, hoechstens);
  const letzte = gekappt[hoechstens - 1];
  gekappt[hoechstens - 1] = `${letzte.slice(0, Math.max(0, zeichen - 1)).trimEnd()}…`;
  return gekappt;
}

/** Welche Bedeutung eine Folge trägt — dieselben drei wie überall sonst. */
export function folgenArt(folge: Folge): 'agree' | 'contra' | 'unique' {
  if (folge.gegensatz.length > 0) return 'contra';
  return folge.anzahl > 1 ? 'agree' : 'unique';
}

const FOLGE_ZEICHEN = 40;
const FOLGE_ZEILEN = 4;

/**
 * Die Konsequenzkarte einer Szenario-Runde.
 *
 * Links die Ausgangsaussage, rechts die Folgen, die die Stimmen genannt haben.
 * Gewicht trägt eine Folge **ausschließlich** über die Zahl der Stimmen, die
 * sie genannt haben: so viele gefüllte Punkte, wie Stimmen sie nannten, von so
 * vielen Punkten, wie Stimmen geantwortet haben. Das ist eine Häufigkeit und
 * keine Wahrscheinlichkeit — die Beschriftung sagt das auch so.
 */
export function renderSzenario(
  szenario: Szenario | null,
  onSelect: (auswahl: GraphSelection, folge: Folge) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph szenario');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Genannte Folgen eines Szenarios');

  if (!szenario) {
    leereFlaeche(svg, 380, 140,
      'Noch kein Szenario durchgespielt.');
    return svg;
  }
  if (szenario.folgen.length === 0) {
    leereFlaeche(svg, 380, 140, 'Keine auswertbare Folge genannt.');
    return svg;
  }

  const breite = 380;
  const xAusgang = 8;
  const bAusgang = 96;
  const xFolge = 128;
  const bFolge = breite - xFolge - 8;
  const zeilenHoehe = 13;
  const kopfHoehe = 18;
  const abstand = 12;
  const rand = 10;

  // Jede Folge bekommt so viel Platz, wie ihr Wortlaut braucht.
  const gesetzt = szenario.folgen.map((folge) => ({
    folge,
    zeilen: umbruch(folge.text, FOLGE_ZEICHEN, FOLGE_ZEILEN),
  }));
  const hoehen = gesetzt.map((g) => kopfHoehe + g.zeilen.length * zeilenHoehe + 8);
  const fussraum = szenario.uebergangen > 0 ? 14 : 0;
  const hoehe = rand * 2 + fussraum + hoehen.reduce((a, b) => a + b, 0)
    + abstand * Math.max(0, gesetzt.length - 1);
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  const yOben: number[] = [];
  let lauf = rand;
  for (const h of hoehen) {
    yOben.push(lauf);
    lauf += h + abstand;
  }
  const mitteAusgang = (hoehe - fussraum) / 2;

  // Fäden zuerst: die Kästen liegen darüber und bleiben lesbar.
  gesetzt.forEach((g, i) => {
    const y = yOben[i] + hoehen[i] / 2;
    const x1 = xAusgang + bAusgang;
    const mx = (x1 + xFolge) / 2;
    const pfad = document.createElementNS(SVG_NS, 'path');
    pfad.setAttribute('d', `M ${x1} ${mitteAusgang} C ${mx} ${mitteAusgang}, ${mx} ${y}, ${xFolge} ${y}`);
    pfad.setAttribute('class', `faden ${folgenArt(g.folge)}`);
    // Die Strichbreite trägt die Zahl der Stimmen — nichts sonst.
    pfad.setAttribute('stroke-width', String(Math.min(1.4 + g.folge.anzahl * 0.8, 5)));
    svg.appendChild(pfad);
  });

  // --- Die Ausgangsaussage.
  const ausgang = document.createElementNS(SVG_NS, 'g');
  ausgang.setAttribute('class', 'ausgang');
  const hAusgang = 62;
  const kasten = document.createElementNS(SVG_NS, 'rect');
  kasten.setAttribute('x', String(xAusgang));
  kasten.setAttribute('y', String(mitteAusgang - hAusgang / 2));
  kasten.setAttribute('width', String(bAusgang));
  kasten.setAttribute('height', String(hAusgang));
  kasten.setAttribute('rx', '8');
  ausgang.appendChild(kasten);

  const name = szenario.ausgang?.label ?? 'deine Eingabe';
  const kreis = document.createElementNS(SVG_NS, 'circle');
  kreis.setAttribute('cx', String(xAusgang + bAusgang / 2));
  kreis.setAttribute('cy', String(mitteAusgang - hAusgang / 2 + 18));
  kreis.setAttribute('r', '12');
  kreis.setAttribute('class', 'ausgangssignet');
  ausgang.appendChild(kreis);
  ausgang.appendChild(
    text(xAusgang + bAusgang / 2, mitteAusgang - hAusgang / 2 + 22, initialen(name)),
  );
  ausgang.appendChild(
    text(xAusgang + bAusgang / 2, mitteAusgang - hAusgang / 2 + 44, shorten(name), 'zahl'),
  );
  ausgang.appendChild(
    text(xAusgang + bAusgang / 2, mitteAusgang - hAusgang / 2 + 56, 'sagte', 'leer'),
  );
  const titelAusgang = document.createElementNS(SVG_NS, 'title');
  titelAusgang.textContent = szenario.ausgang
    ? `${szenario.ausgang.label}: ${szenario.ausgang.auszug}`
    : `Deine Eingabe: ${szenario.prompt}`;
  ausgang.appendChild(titelAusgang);
  svg.appendChild(ausgang);

  // --- Die genannten Folgen.
  gesetzt.forEach((g, i) => {
    const { folge, zeilen } = g;
    const y = yOben[i];
    const art = folgenArt(folge);
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', `folge ${art}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.dataset.folge = folge.id;

    const flaeche = document.createElementNS(SVG_NS, 'rect');
    flaeche.setAttribute('x', String(xFolge));
    flaeche.setAttribute('y', String(y));
    flaeche.setAttribute('width', String(bFolge));
    flaeche.setAttribute('height', String(hoehen[i]));
    flaeche.setAttribute('rx', '7');
    flaeche.setAttribute('class', 'folgenflaeche');
    gruppe.appendChild(flaeche);

    // Die Auszählung: ein Punkt je Stimme, gefüllt für jede, die es nannte.
    for (let k = 0; k < folge.von; k += 1) {
      const punkt = document.createElementNS(SVG_NS, 'circle');
      punkt.setAttribute('cx', String(xFolge + 10 + k * 9));
      punkt.setAttribute('cy', String(y + 11));
      punkt.setAttribute('r', '3.4');
      punkt.setAttribute('class', `stimmpunkt ${k < folge.anzahl ? `voll ${art}` : 'leer'}`);
      gruppe.appendChild(punkt);
    }
    const haeufigkeit = `von ${folge.anzahl} von ${folge.von} Stimmen genannt`;
    gruppe.appendChild(
      text(xFolge + 16 + folge.von * 9, y + 14.5, haeufigkeit, 'haeufigkeit', 'start'),
    );

    zeilen.forEach((zeile, z) => {
      gruppe.appendChild(
        text(xFolge + 10, y + kopfHoehe + 10 + z * zeilenHoehe, zeile, 'folgentext', 'start'),
      );
    });

    const namen = [...new Set(folge.nennungen.map((n) => n.label))];
    const gegen = folge.gegensatz.length > 0
      ? ' — steht einer anderen genannten Folge entgegen'
      : '';
    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent = `${haeufigkeit} (${namen.join(', ')})${gegen}\n${folge.text}`;
    gruppe.appendChild(titel);
    gruppe.setAttribute(
      'aria-label',
      `Folge, ${haeufigkeit}${gegen}: ${folge.text} — Antworten öffnen`,
    );

    bedienbar(gruppe, () =>
      onSelect(
        {
          jobIds: [...new Set(folge.nennungen.map((n) => n.job_id))],
          label: `Folge, ${haeufigkeit}`,
        },
        folge,
      ),
    );
    svg.appendChild(gruppe);
  });

  if (szenario.uebergangen > 0) {
    svg.appendChild(
      text(breite / 2, hoehe - 4, `+ ${szenario.uebergangen} weitere genannte Folgen`, 'leer'),
    );
  }

  return svg;
}

// --------------------------------------------------------- Herkunfts-Linse

const HERKUNFT_BAENDER: { status: Relation['status']; titel: string; satz: string }[] = [
  {
    status: 'bestaetigt',
    titel: 'Von dir bestätigt',
    satz: 'Darauf kannst du dich berufen.',
  },
  {
    status: 'vorschlag',
    titel: 'Vorschlag der Auswertung',
    satz: 'Noch von niemandem festgestellt.',
  },
  {
    status: 'abgelehnt',
    titel: 'Von dir verworfen',
    satz: 'Bleibt sichtbar, damit die Entscheidung nachvollziehbar bleibt.',
  },
];

const BEZUGS_TEXT: Record<string, string> = {
  antwortet_auf: 'antwortet auf',
  abgeleitet_aus: 'abgeleitet aus',
  widerspricht: 'widerspricht',
  uebereinstimmung: 'stimmt überein mit',
  vertieft: 'vertieft',
};

/** Welche Bedeutung ein Bezug trägt. Ohne Befund bleibt er farblos. */
function bezugsArt(typ: string): 'agree' | 'contra' | 'neutral' {
  if (typ === 'widerspricht') return 'contra';
  if (typ === 'uebereinstimmung') return 'agree';
  return 'neutral';
}

/**
 * Was der Mensch festgestellt hat — und was bloß ein Vorschlag ist.
 *
 * In einem Werkzeug, das keine Wahrheit behauptet, ist das die Linse, die
 * „worauf kann ich mich hier berufen?" beantwortet. Der Stand steht in der
 * **Strichart**: bestätigt durchgezogen, Vorschlag gestrichelt, verworfen
 * ausgegraut und gepunktet. Die **Herkunft** steht in der Form am Faden:
 * eine gefüllte Raute, wo du selbst gesetzt hast, ein offener Ring, wo die
 * Auswertung vorgeschlagen hat. Farbe bleibt dem Befund vorbehalten.
 */
export function renderHerkunft(
  relations: Relation[],
  namen: Map<string, string>,
  onSelect: (auswahl: GraphSelection) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph herkunft');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Herkunft und Stand der gesetzten Bezüge');

  if (relations.length === 0) {
    leereFlaeche(svg, 380, 120,
      'Noch kein Bezug gesetzt — es gibt hier nichts, worauf du dich berufen könntest.');
    return svg;
  }

  const breite = 380;
  const zeile = 30;
  const bandKopf = 26;
  const rand = 8;
  const gruppiert = HERKUNFT_BAENDER.map((band) => ({
    ...band,
    bezuege: relations.filter((r) => r.status === band.status),
  }));

  let hoehe = rand;
  const yBand: number[] = [];
  for (const band of gruppiert) {
    yBand.push(hoehe);
    hoehe += bandKopf + Math.max(1, band.bezuege.length) * zeile + 6;
  }
  hoehe += rand;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  gruppiert.forEach((band, i) => {
    const y = yBand[i];
    svg.appendChild(
      text(8, y + 12, `${band.titel} — ${band.bezuege.length}`, 'bandkopf', 'start'),
    );
    if (band.bezuege.length === 0) {
      svg.appendChild(text(8, y + bandKopf + 14, '— keiner —', 'leer', 'start'));
      return;
    }

    band.bezuege.forEach((bezug, k) => {
      const zy = y + bandKopf + k * zeile + 14;
      const gruppe = document.createElementNS(SVG_NS, 'g');
      gruppe.setAttribute('class', `bezug ${band.status} ${bezugsArt(bezug.type)}`);
      gruppe.setAttribute('tabindex', '0');
      gruppe.setAttribute('role', 'button');
      gruppe.dataset.relation = bezug.id;

      const von = namen.get(bezug.from_id) ?? bezug.from_id;
      const nach = namen.get(bezug.to_id) ?? bezug.to_id;
      const x1 = 124;
      const x2 = 252;

      const flaeche = document.createElementNS(SVG_NS, 'rect');
      flaeche.setAttribute('x', '4');
      flaeche.setAttribute('y', String(zy - 12));
      flaeche.setAttribute('width', String(breite - 8));
      flaeche.setAttribute('height', '24');
      flaeche.setAttribute('rx', '6');
      flaeche.setAttribute('class', 'bezugflaeche');
      gruppe.appendChild(flaeche);

      const strich = document.createElementNS(SVG_NS, 'path');
      strich.setAttribute('d', `M ${x1} ${zy} H ${x2}`);
      strich.setAttribute('class', 'bezugstrich');
      gruppe.appendChild(strich);

      // Die Herkunft als Form in der Mitte des Fadens, nie als eigene Farbe.
      const mx = (x1 + x2) / 2;
      const zeichen = document.createElementNS(SVG_NS, 'path');
      zeichen.setAttribute(
        'd',
        bezug.origin === 'mensch'
          ? `M ${mx} ${zy - 5} L ${mx + 5} ${zy} L ${mx} ${zy + 5} L ${mx - 5} ${zy} Z`
          : `M ${mx - 4.5} ${zy} a 4.5 4.5 0 1 0 9 0 a 4.5 4.5 0 1 0 -9 0`,
      );
      zeichen.setAttribute('class', `herkunftszeichen ${bezug.origin}`);
      gruppe.appendChild(zeichen);

      gruppe.appendChild(text(x1 - 6, zy + 4, shorten(von), 'bezugname', 'end'));
      gruppe.appendChild(text(x2 + 6, zy + 4, shorten(nach), 'bezugname', 'start'));
      gruppe.appendChild(
        text(mx, zy - 9, BEZUGS_TEXT[bezug.type] ?? bezug.type, 'bezugart'),
      );

      const herkunftswort = bezug.origin === 'mensch'
        ? 'von dir gesetzt'
        : 'maschineller Vorschlag';
      const standwort = band.titel.toLowerCase();
      const titel = document.createElementNS(SVG_NS, 'title');
      titel.textContent =
        `${von} ${BEZUGS_TEXT[bezug.type] ?? bezug.type} ${nach}\n` +
        `${herkunftswort}, ${standwort}` + (bezug.note ? `\n${bezug.note}` : '');
      gruppe.appendChild(titel);
      gruppe.setAttribute(
        'aria-label',
        `${von} ${BEZUGS_TEXT[bezug.type] ?? bezug.type} ${nach}, ` +
          `${herkunftswort}, ${standwort} — Antworten öffnen`,
      );

      bedienbar(gruppe, () =>
        onSelect({
          jobIds: [bezug.from_id, bezug.to_id],
          label: `${von} ${BEZUGS_TEXT[bezug.type] ?? bezug.type} ${nach}`,
        }),
      );
      svg.appendChild(gruppe);
    });
  });

  return svg;
}

// -------------------------------------------------------------- Zeitlinse

/** Sekunden lesbar machen, ohne etwas zu runden, was nicht gemessen wurde. */
export function dauerText(sekunden: number): string {
  if (sekunden < 1) return `${Math.round(sekunden * 1000)} ms`;
  return `${sekunden.toFixed(1)} s`;
}

/**
 * Eine Runde als Zeitbild.
 *
 * Je Auftrag zwei Abschnitte: das Warten in der Schlange des Anbieters und das
 * Schreiben. Erst dadurch wird sichtbar, dass eine Karte nicht langsam war,
 * sondern lange gewartet hat — die Warteschlange je Anbieter hatte bisher kein
 * Bild.
 *
 * Die Zeilen stehen in der Reihenfolge des Tisches und **nicht** nach Dauer
 * sortiert: es entsteht keine Rangfolge. Gezeigt wird nur, was gemessen wurde;
 * wo eine Zeit fehlt, steht kein Balken.
 */
export function renderZeit(
  jobs: Job[],
  onSelect: (auswahl: GraphSelection) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph zeit');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Zeitlicher Ablauf der Runde');

  const zeiten = jobs.filter((j) => Number.isFinite(j.created_at));
  const enden = zeiten
    .flatMap((j) => [j.started_at, j.finished_at])
    .filter((t): t is number => typeof t === 'number' && Number.isFinite(t));
  if (zeiten.length === 0 || enden.length === 0) {
    leereFlaeche(svg, 380, 120, 'Noch keine Zeiten erfasst.');
    return svg;
  }

  const beginn = Math.min(...zeiten.map((j) => j.created_at));
  const ende = Math.max(...enden, beginn);
  // Eine Runde ohne messbare Dauer bekommt trotzdem eine Skala, damit die
  // Balken nicht durch Null geteilt werden.
  const spanne = Math.max(ende - beginn, 0.001);

  const breite = 380;
  const xAchse = 76;
  const bAchse = breite - xAchse - 10;
  const zeile = 26;
  const oben = 22;
  const hoehe = oben + jobs.length * zeile + 8;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  const x = (t: number) => xAchse + ((t - beginn) / spanne) * bAchse;

  // --- Achse: vier Marken über die gemessene Spanne.
  for (let i = 0; i <= 4; i += 1) {
    const anteil = i / 4;
    const px = xAchse + anteil * bAchse;
    const linie = document.createElementNS(SVG_NS, 'path');
    linie.setAttribute('d', `M ${px} ${oben - 8} V ${oben + jobs.length * zeile - 6}`);
    linie.setAttribute('class', 'zeitgitter');
    svg.appendChild(linie);
    // Die äußeren Marken werden nach innen gesetzt, sonst stünde ihre Hälfte
    // außerhalb der Fläche und wäre abgeschnitten.
    const anker = i === 0 ? 'start' : i === 4 ? 'end' : 'middle';
    svg.appendChild(
      text(px, oben - 12, dauerText(spanne * anteil), 'zeitmarke', anker),
    );
  }

  jobs.forEach((job, i) => {
    const y = oben + i * zeile;
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', `zeitzeile zustand-${job.status}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.dataset.jobId = job.id;

    const flaeche = document.createElementNS(SVG_NS, 'rect');
    flaeche.setAttribute('x', '2');
    flaeche.setAttribute('y', String(y - 2));
    flaeche.setAttribute('width', String(breite - 4));
    flaeche.setAttribute('height', String(zeile - 4));
    flaeche.setAttribute('rx', '6');
    flaeche.setAttribute('class', 'zeitflaeche');
    gruppe.appendChild(flaeche);

    gruppe.appendChild(text(xAchse - 8, y + 13, shorten(job.label), 'zeitname', 'end'));

    const start = typeof job.started_at === 'number' ? job.started_at : null;
    const schluss = typeof job.finished_at === 'number' ? job.finished_at : null;
    const teile: string[] = [];

    if (start !== null && start > job.created_at) {
      const warten = document.createElementNS(SVG_NS, 'rect');
      warten.setAttribute('x', String(x(job.created_at)));
      warten.setAttribute('y', String(y + 7));
      warten.setAttribute('width', String(Math.max(1, x(start) - x(job.created_at))));
      warten.setAttribute('height', '3');
      warten.setAttribute('class', 'warteband');
      gruppe.appendChild(warten);
      teile.push(`${dauerText(start - job.created_at)} gewartet`);
    }

    if (start !== null && schluss !== null && schluss >= start) {
      const schreiben = document.createElementNS(SVG_NS, 'rect');
      schreiben.setAttribute('x', String(x(start)));
      schreiben.setAttribute('y', String(y + 3));
      schreiben.setAttribute('width', String(Math.max(2, x(schluss) - x(start))));
      schreiben.setAttribute('height', '11');
      schreiben.setAttribute('rx', '2');
      schreiben.setAttribute('class', `schreibband zustand-${job.status}`);
      gruppe.appendChild(schreiben);
      teile.push(`${dauerText(schluss - start)} geschrieben`);
    } else if (start !== null) {
      // Läuft noch: ein offener Strich statt eines erfundenen Endes.
      const offen = document.createElementNS(SVG_NS, 'path');
      offen.setAttribute('d', `M ${x(start)} ${y + 8.5} H ${xAchse + bAchse}`);
      offen.setAttribute('class', 'offenesband');
      gruppe.appendChild(offen);
      teile.push('läuft noch');
    } else {
      teile.push('nicht begonnen');
    }

    const satz = `${job.label}: ${teile.join(', ')}`;
    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent = satz;
    gruppe.appendChild(titel);
    gruppe.setAttribute('aria-label', `${satz} — Antwort öffnen`);

    bedienbar(gruppe, () => onSelect({ jobIds: [job.id], label: job.label }));
    svg.appendChild(gruppe);
  });

  // Der erklärende Satz steht als Fließtext neben der Linse, nicht im Bild:
  // im SVG bricht er nicht um und stünde über den Rand hinaus.
  svg.setAttribute(
    'aria-label',
    'Zeitlicher Ablauf der Runde — gemessene Zeiten in der Reihenfolge des Tisches, '
      + 'keine Rangfolge',
  );
  return svg;
}
