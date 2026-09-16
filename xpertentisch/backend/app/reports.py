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

from .analysis import KIND_AGREEMENT, KIND_CONTRADICTION, KIND_UNIQUE

KIND_LABELS = {
    KIND_AGREEMENT: "Übereinstimmung",
    KIND_CONTRADICTION: "Widerspruch",
    KIND_UNIQUE: "Einzigartig",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_time(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d.%m.%Y %H:%M:%S UTC")


def _status_label(job: dict[str, Any]) -> str:
    return {
        "done": "fertig",
        "error": "Fehler",
        "interrupted": "unterbrochen",
        "queued": "wartet",
        "running": "läuft",
    }.get(job["status"], job["status"])


REPORT_CSS = """
:root {
  color-scheme: light;
  --ink: #1b1f2a; --muted: #5b6478; --line: #d8dde8; --bg: #f6f7fb;
  --card: #ffffff; --agree: #1f7a4d; --contra: #b03a2e; --unique: #6b4fa8;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--ink);
  font: 16px/1.6 "Iowan Old Style", "Palatino Linotype", Georgia, serif;
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 1.9rem; margin: 0 0 4px; letter-spacing: .02em; }
h2 { font-size: 1.3rem; margin: 40px 0 12px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 1.05rem; margin: 24px 0 8px; }
.meta { color: var(--muted); font-size: .9rem; margin-bottom: 24px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 18px 20px; margin: 14px 0; }
.answer { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; }
.badge { display: inline-block; font-size: .78rem; padding: 2px 9px; border-radius: 999px;
         border: 1px solid var(--line); color: var(--muted); margin-right: 6px; font-family: system-ui, sans-serif; }
.badge.err { color: var(--contra); border-color: var(--contra); }
.badge.part { color: #8a6d1f; border-color: #c8a43a; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: .92rem;
        font-family: system-ui, -apple-system, sans-serif; }
th, td { border: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: #eef1f7; }
.tablewrap { overflow-x: auto; }
blockquote { margin: 6px 0 6px 0; padding: 6px 12px; border-left: 3px solid var(--line);
             color: var(--ink); background: #fbfcfe; }
.k-uebereinstimmung { border-left-color: var(--agree); }
.k-widerspruch { border-left-color: var(--contra); }
.k-einzigartig { border-left-color: var(--unique); }
.kind { font-family: system-ui, sans-serif; font-size: .8rem; text-transform: uppercase;
        letter-spacing: .06em; color: var(--muted); }
.note { color: var(--muted); font-size: .85rem; font-family: system-ui, sans-serif; }
details { margin: 8px 0; }
summary { cursor: pointer; font-family: system-ui, sans-serif; font-size: .92rem; }
footer { margin-top: 48px; color: var(--muted); font-size: .82rem; }
@media (max-width: 640px) {
  body { padding: 18px 12px 48px; font-size: 15px; }
  h1 { font-size: 1.45rem; }
}
@media print {
  body { background: #fff; padding: 0; font-size: 11pt; }
  .card { break-inside: avoid; border-color: #bbb; }
  h2 { break-after: avoid; }
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
                a('<p class="note">Keine Antwort erfasst.</p>')

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
            job_markers = [m for m in entry["markers"] if m["job_id"] == job["id"]]
            if job_markers:
                a("**Marker**")
                a("")
                for m in job_markers:
                    a(f"- _{KIND_LABELS.get(m['kind'], m['kind'])}_: „{m['quote']}“ — {m['note']}")
                a("")

    a("## Hinweise")
    a("")
    a("Die Antworten sind unverändert wiedergegeben. Marker stammen aus einem "
      "regelbasierten Textvergleich ohne weitere Modellaufrufe; sie sind Lesehilfen, "
      "keine Bewertung und keine Rangfolge der Modelle.")
    a("")
    return "\n".join(out)
