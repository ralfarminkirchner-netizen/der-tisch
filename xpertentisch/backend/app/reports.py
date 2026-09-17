"""Berichtsexporte: eigenständiges HTML und Markdown.

Das HTML ist offline nutzbar (keine externen Ressourcen, kein Skript),
druckbar (eigene @media print-Regeln) und gegen Injection geschützt: jeder
Inhalt aus Antworten, Fragen oder Fehlermeldungen wird escaped. Auf
JavaScript wird bewusst vollständig verzichtet — Aufklappen geschieht über
<details>. Die Originalantworten werden unverändert übernommen.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .analysis import KIND_AGREEMENT, KIND_CONTRADICTION, KIND_UNIQUE

KIND_LABELS = {
    KIND_AGREEMENT: "Übereinstimmung",
    KIND_CONTRADICTION: "Widerspruch",
    KIND_UNIQUE: "Einzigartig",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


#: Gespeichert wird in UTC, gezeigt wird die Zeit, in der gearbeitet wurde.
ANZEIGEZONE = ZoneInfo("Europe/Berlin")


def fmt_time(ts: float | None) -> str:
    if not ts:
        return "—"
    lokal = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ANZEIGEZONE)
    return lokal.strftime("%d.%m.%Y %H:%M:%S %Z")


def _status_label(job: dict[str, Any]) -> str:
    return {
        "done": "fertig",
        "error": "Fehler",
        "interrupted": "unterbrochen",
        "queued": "wartet",
        "running": "läuft",
        "not_requested": "nicht angefragt",
        "cancelled": "abgebrochen",
    }.get(job["status"], job["status"])


BEZIEHUNG_TEXT = {
    "antwortet_auf": "antwortet auf",
    "abgeleitet_aus": "abgeleitet aus",
    "widerspricht": "widerspricht",
    "uebereinstimmung": "stimmt überein mit",
    "vertieft": "vertieft",
}

HERKUNFT_TEXT = {
    "mensch": "von dir gesetzt",
    "maschine": "maschineller Vorschlag",
}

STAND_TEXT = {
    "vorschlag": "unbestätigt",
    "bestaetigt": "von dir bestätigt",
    "abgelehnt": "von dir verworfen",
}


def _leer_grund(job: dict[str, Any]) -> str:
    """Warum hier kein Text steht — die Fälle sind nicht dasselbe."""
    return {
        "not_requested": "Für diesen Funken nicht angefragt.",
        "cancelled": "Abgebrochen, bevor eine Antwort kam.",
        "done": "Antwort kam an, enthielt aber keinen Text.",
        "queued": "Wartet noch.",
        "running": "Läuft noch.",
    }.get(job["status"], "Keine Antwort erfasst.")


def _beitragsnamen(bundle: dict[str, Any]) -> dict[str, str]:
    namen: dict[str, str] = {}
    for entry in bundle["sparks"]:
        namen[entry["spark"]["id"]] = f"Funke {entry['spark']['seq']} (deine Eingabe)"
        for job in entry["jobs"]:
            namen[job["id"]] = f"{job['label']} zu Funke {entry['spark']['seq']}"
    return namen


REPORT_CSS = """
/* Der Bericht ist ein Dokument, kein Abzug der Oberfläche.

   Er soll sich lesen wie ein sauber gesetztes Protokoll: ein ruhiges Maß,
   Serife im Fließtext, Grotesk für alles Technische, Haarlinien statt Kästen.
   Er lädt nichts nach — auch keine Schrift. Die Anwendung bringt eine eigene
   Schrift mit; sie hier als Base64 einzubetten würde die Datei um ein
   Vielfaches aufblähen, ohne dass der Bericht davon lesbarer würde. Er nutzt
   darum einen Systemschrift-Stapel, der dieselbe Gliederung trägt. */

:root {
  color-scheme: light dark;
  --ink: #1a1815;
  --muted: #6a645c;
  --leise: #8c857b;
  --line: #ddd8ce;
  --bg: #f7f5f1;
  --card: #fffdfa;
  --agree: #1c6b45;
  --contra: #a63d1c;
  --unique: #5b3f9e;
  --warn: #8a6415;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: clamp(28px, 6vw, 72px) clamp(16px, 5vw, 40px) 80px;
  background: var(--bg);
  color: var(--ink);
  font: 17px/1.62 var(--serif);
  -webkit-text-size-adjust: 100%;
}

.wrap { max-width: 44rem; margin: 0 auto; }

/* Titelblock: der Bericht sagt zuerst, was er ist. */
h1 {
  font-size: clamp(1.7rem, 1.2rem + 2vw, 2.5rem);
  line-height: 1.12;
  margin: 0 0 6px;
  font-weight: 600;
  letter-spacing: -0.015em;
}

h2 {
  font-family: var(--sans);
  font-size: .78rem;
  font-weight: 650;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--leise);
  margin: 52px 0 14px;
  padding-bottom: 7px;
  border-bottom: 1px solid var(--line);
}

h3 {
  font-size: 1.08rem;
  font-weight: 600;
  margin: 0 0 6px;
  letter-spacing: -0.005em;
}

p { margin: 0 0 .7em; }
p:last-child { margin-bottom: 0; }

.meta {
  font-family: var(--sans);
  color: var(--muted);
  font-size: .82rem;
  margin: 0 0 8px;
  padding-bottom: 22px;
  border-bottom: 2px solid var(--ink);
}

/* Ein Beitrag steht auf eigenem Grund, aber ohne Kastencharakter. */
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 18px 22px;
  margin: 16px 0;
}

.answer {
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  hyphens: auto;
}

.badge {
  display: inline-block;
  font-family: var(--sans);
  font-size: .72rem;
  font-weight: 560;
  letter-spacing: .02em;
  padding: 2px 9px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  margin: 0 6px 4px 0;
  white-space: nowrap;
}

.badge.err { color: var(--contra); border-color: var(--contra); }
.badge.part { color: var(--warn); border-color: var(--warn); }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
  font-family: var(--sans);
  font-size: .82rem;
  font-variant-numeric: tabular-nums;
}

th, td {
  border: 0;
  border-bottom: 1px solid var(--line);
  padding: 8px 10px 8px 0;
  text-align: right;
  vertical-align: top;
  white-space: nowrap;
}

th {
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--leise);
  border-bottom: 1px solid var(--ink);
}

th:first-child, td:first-child {
  text-align: left;
  white-space: normal;
}

tbody tr:last-child td { border-bottom: 0; }

/* Die Vergleichstabelle hat neun Spalten und passt nicht in das Maß des
   Fließtextes. Sie darf darum aus dem Satzspiegel heraustreten — aber nur so
   weit, wie der Bildschirm es hergibt, damit keine Überbreite entsteht. */
.tablewrap {
  overflow-x: auto;
  max-width: 100%;
  margin-inline: calc(-1 * clamp(0px, (100vw - 44rem) / 2 - 1rem, 7rem));
}

/* Ein Marker ist ein Beleg am Rand, keine Behauptung im Text. */
blockquote {
  margin: 10px 0;
  padding: 2px 0 2px 14px;
  border-left: 2px solid var(--line);
  color: var(--ink);
  font-size: .95rem;
}

.k-uebereinstimmung { border-left-color: var(--agree); }
.k-widerspruch { border-left-color: var(--contra); }
.k-einzigartig { border-left-color: var(--unique); }

.kind {
  font-family: var(--sans);
  font-size: .68rem;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--leise);
  margin-bottom: 2px;
}

.k-uebereinstimmung .kind { color: var(--agree); }
.k-widerspruch .kind { color: var(--contra); }
.k-einzigartig .kind { color: var(--unique); }

.note {
  color: var(--muted);
  font-size: .82rem;
  font-family: var(--sans);
  line-height: 1.55;
}

details { margin: 10px 0 0; }

summary {
  cursor: pointer;
  font-family: var(--sans);
  font-size: .8rem;
  color: var(--muted);
}

footer {
  margin-top: 56px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  color: var(--leise);
  font-size: .78rem;
  font-family: var(--sans);
}

/* Auch ein Dokument wird abends gelesen. Gedruckt wird es trotzdem hell —
   die Druckregeln unten setzen den Grund wieder auf Weiß. */
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #eae5dd;
    --muted: #b0a89c;
    --leise: #8f877b;
    --line: #3a352e;
    --bg: #17150f;
    --card: #1e1b16;
    --agree: #74d3a0;
    --contra: #f0a07a;
    --unique: #b9a2f0;
    --warn: #dcc07a;
  }
}

@media (max-width: 640px) {
  body { font-size: 16px; }
}

@media print {
  /* Auf Papier gilt immer der helle Satz, unabhängig vom Gerät. */
  :root {
    --ink: #1a1815;
    --muted: #55504a;
    --leise: #6f6961;
    --line: #c9c3b8;
    --bg: #fff;
    --card: #fff;
    --agree: #1c6b45;
    --contra: #a63d1c;
    --unique: #5b3f9e;
    --warn: #8a6415;
  }

  body { background: #fff; color: var(--ink); padding: 0; font-size: 10.5pt; }
  .wrap { max-width: none; }

  /* Kurze Blöcke bleiben zusammen. Eine lange Antwort wird NICHT auf eine
     Seite gezwungen — sie bräche sonst lieber ganz ab, als umzubrechen. */
  .card { break-inside: auto; border-color: #bbb; background: #fff; }
  blockquote, tr, .meta { break-inside: avoid; }
  h1, h2, h3 { break-after: avoid; }
  details { display: block; }
  details > summary { display: none; }
  footer { position: static; }
}
"""


def render_html(bundle: dict[str, Any]) -> str:
    session = bundle["session"]
    parts: list[str] = []
    a = parts.append

    a("<!DOCTYPE html>")
    a('<html lang="de"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width, initial-scale=1">')
    a(f"<title>XPERTENTiSCH — {esc(session['title'])}</title>")
    a(f"<style>{REPORT_CSS}</style>")
    a("</head><body><div class=\"wrap\">")
    a(f"<h1>XPERTENTiSCH — {esc(session['title'])}</h1>")
    a(
        '<p class="meta">Sitzung {sid} · begonnen {start} · Status {status}{closed}</p>'.format(
            sid=esc(session["id"]),
            start=esc(fmt_time(session["created_at"])),
            status=esc(session["status"]),
            closed=(
                f" · abgeschlossen {esc(fmt_time(session['closed_at']))}"
                if session.get("closed_at")
                else ""
            ),
        )
    )
    if bundle.get("pending"):
        a('<div class="card"><h3>Vorläufiger Stand</h3><p class="note">'
          "Zum Zeitpunkt dieses Berichts liefen noch Aufträge. Er bildet also "
          "einen Zwischenstand ab, keine vollständige Runde. Ein später erzeugter "
          "Bericht kann mehr enthalten.</p></div>")
    if session.get("closing_note"):
        a(f'<div class="card"><h3>Abschlussnotiz</h3><div class="answer">{esc(session["closing_note"])}</div></div>')

    for entry in bundle["sparks"]:
        spark = entry["spark"]
        a(f"<h2>Funke {esc(spark['seq'])}</h2>")
        a(f'<div class="card"><strong>Frage</strong><div class="answer">{esc(spark["prompt"])}</div>'
          f'<p class="note">gestellt {esc(fmt_time(spark["created_at"]))}</p></div>')

        summary = entry.get("summary") or {}
        rows = summary.get("models") or []
        if rows:
            a("<h3>Vergleich</h3><div class=\"tablewrap\"><table><thead><tr>"
              "<th>Modell</th><th>Provider</th><th>Status</th><th>Zeichen</th><th>Sätze</th>"
              "<th>Dauer</th><th>Übereinstimmungen</th><th>Widersprüche</th><th>Einzigartig</th>"
              "</tr></thead><tbody>")
            for r in rows:
                dur = f"{r['latency_ms']} ms" if r.get("latency_ms") else "—"
                a("<tr>"
                  f"<td>{esc(r['label'])}</td><td>{esc(r['provider'])}</td>"
                  f"<td>{esc(r['status'])}</td><td>{esc(r['chars'])}</td>"
                  f"<td>{esc(r['sentences'])}</td><td>{esc(dur)}</td>"
                  f"<td>{esc(r['agreements'])}</td><td>{esc(r['contradictions'])}</td>"
                  f"<td>{esc(r['unique'])}</td></tr>")
            a("</tbody></table></div>")

        for job in entry["jobs"]:
            badges = [f'<span class="badge">{esc(job["provider"])} · {esc(job["model"])}</span>']
            if job["status"] == "error":
                badges.append('<span class="badge err">Fehler</span>')
            if job["status"] == "interrupted":
                badges.append('<span class="badge err">unterbrochen</span>')
            if job.get("partial"):
                badges.append('<span class="badge part">Teilantwort</span>')
            a(f'<div class="card"><h3>{esc(job["label"])}</h3>')
            a(f'<p>{"".join(badges)}<span class="badge">{esc(_status_label(job))}</span></p>')
            if job.get("error"):
                a(f'<p class="note">Fehlermeldung: {esc(job["error"])}</p>')
            text = job.get("text") or ""
            if text.strip():
                a(f'<div class="answer">{esc(text)}</div>')
            elif not job.get("error"):
                a(f'<p class="note">{esc(_leer_grund(job))}</p>')

            job_markers = [m for m in entry["markers"] if m["job_id"] == job["id"]]
            if job_markers:
                a(f"<details open><summary>{len(job_markers)} Marker</summary>")
                for m in job_markers:
                    a(f'<blockquote class="k-{esc(m["kind"])}">'
                      f'<div class="kind">{esc(KIND_LABELS.get(m["kind"], m["kind"]))}</div>'
                      f'{esc(m["quote"])}'
                      f'<div class="note">{esc(m["note"])}</div></blockquote>')
                a("</details>")
            a("</div>")

    beziehungen = bundle.get("relations") or []
    if beziehungen:
        namen = _beitragsnamen(bundle)
        a("<h2>Bezüge</h2>")
        a('<div class="tablewrap"><table><thead><tr><th>Von</th><th>Art</th><th>Zu</th>'
          "<th>Herkunft</th><th>Stand</th></tr></thead><tbody>")
        for bez in beziehungen:
            a("<tr>"
              f"<td>{esc(namen.get(bez['from_id'], bez['from_id']))}</td>"
              f"<td>{esc(BEZIEHUNG_TEXT.get(bez['type'], bez['type']))}</td>"
              f"<td>{esc(namen.get(bez['to_id'], bez['to_id']))}</td>"
              f"<td>{esc(HERKUNFT_TEXT.get(bez['origin'], bez['origin']))}</td>"
              f"<td>{esc(STAND_TEXT.get(bez['status'], bez['status']))}</td></tr>")
        a("</tbody></table></div>")
        a('<p class="note">Maschinelle Vorschläge sind keine von dir getroffenen '
          "Feststellungen. Was du nicht bestätigt hast, steht als unbestätigt da.</p>")

    a("<h2>Hinweise</h2>")
    a('<div class="card"><p class="note">Die Antworten sind unverändert wiedergegeben. '
      "Marker stammen aus einem regelbasierten Textvergleich ohne weitere Modellaufrufe; "
      "sie sind Lesehilfen, keine Bewertung und keine Rangfolge der Modelle. "
      "Unterschiedliche Aussagen werden nur dann als Widerspruch ausgewiesen, wenn sie "
      "dasselbe Thema betreffen und entgegengesetzte Polarität haben.</p></div>")
    a(f'<footer>XPERTENTiSCH · Bericht erzeugt {esc(fmt_time(bundle["exported_at"]))} · '
      "offline nutzbar, keine externen Ressourcen.</footer>")
    a("</div></body></html>")
    return "\n".join(parts)


def render_markdown(bundle: dict[str, Any]) -> str:
    session = bundle["session"]
    out: list[str] = []
    a = out.append

    a(f"# XPERTENTiSCH — {session['title']}")
    a("")
    a(f"- Sitzung: `{session['id']}`")
    a(f"- Begonnen: {fmt_time(session['created_at'])}")
    a(f"- Status: {session['status']}")
    if session.get("closed_at"):
        a(f"- Abgeschlossen: {fmt_time(session['closed_at'])}")
    a(f"- Bericht erzeugt: {fmt_time(bundle['exported_at'])}")
    if bundle.get("pending"):
        a("- **Vorläufig:** zum Zeitpunkt dieses Berichts liefen noch Aufträge.")
    a("")
    if session.get("closing_note"):
        a("## Abschlussnotiz")
        a("")
        a(session["closing_note"])
        a("")

    for entry in bundle["sparks"]:
        spark = entry["spark"]
        a(f"## Funke {spark['seq']}")
        a("")
        a("**Frage**")
        a("")
        for line in spark["prompt"].splitlines() or [""]:
            a(f"> {line}")
        a("")

        summary = entry.get("summary") or {}
        rows = summary.get("models") or []
        if rows:
            a("| Modell | Provider | Status | Zeichen | Sätze | Dauer | Übereinst. | Widerspr. | Einzigartig |")
            a("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in rows:
                dur = f"{r['latency_ms']} ms" if r.get("latency_ms") else "—"
                a(f"| {r['label']} | {r['provider']} | {r['status']} | {r['chars']} | "
                  f"{r['sentences']} | {dur} | {r['agreements']} | {r['contradictions']} | {r['unique']} |")
            a("")

        for job in entry["jobs"]:
            a(f"### {job['label']} ({job['provider']} · {job['model']})")
            a("")
            flags = [_status_label(job)]
            if job.get("partial"):
                flags.append("Teilantwort")
            a(f"_Status: {', '.join(flags)}_")
            a("")
            if job.get("error"):
                a(f"> Fehler: {job['error']}")
                a("")
            text = (job.get("text") or "").strip()
            if text:
                a(text)
                a("")
            elif not job.get("error"):
                a(f"_{_leer_grund(job)}_")
                a("")
            job_markers = [m for m in entry["markers"] if m["job_id"] == job["id"]]
            if job_markers:
                a("**Marker**")
                a("")
                for m in job_markers:
                    a(f"- _{KIND_LABELS.get(m['kind'], m['kind'])}_: „{m['quote']}“ — {m['note']}")
                a("")

    beziehungen = bundle.get("relations") or []
    if beziehungen:
        namen = _beitragsnamen(bundle)
        a("## Bezüge")
        a("")
        a("| Von | Art | Zu | Herkunft | Stand |")
        a("| --- | --- | --- | --- | --- |")
        for bez in beziehungen:
            a(f"| {namen.get(bez['from_id'], bez['from_id'])} "
              f"| {BEZIEHUNG_TEXT.get(bez['type'], bez['type'])} "
              f"| {namen.get(bez['to_id'], bez['to_id'])} "
              f"| {HERKUNFT_TEXT.get(bez['origin'], bez['origin'])} "
              f"| {STAND_TEXT.get(bez['status'], bez['status'])} |")
        a("")
        a("Maschinelle Vorschläge sind keine von dir getroffenen Feststellungen.")
        a("")

    a("## Hinweise")
    a("")
    a("Die Antworten sind unverändert wiedergegeben. Marker stammen aus einem "
      "regelbasierten Textvergleich ohne weitere Modellaufrufe; sie sind Lesehilfen, "
      "keine Bewertung und keine Rangfolge der Modelle.")
    a("")
    return "\n".join(out)
