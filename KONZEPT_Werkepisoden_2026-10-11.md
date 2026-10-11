# KONZEPT — Werkepisoden in XPERTENTiSCH

**Datum:** 2026-10-11 · **Status:** Konzeptfassung mit erstem Backend-Prototyp
**Arbeitsbereich:** `xpertentisch/` · **Löst nichts ab.** Kein vorhandenes Dokument wurde geändert.

Leitgedanke: **Eine Episode wird abgeschlossen, das Werk bleibt offen.** Der
Entstehungsprozess — Fragen, Entscheidungen, Versuche, Erfahrungen beim Bauen und
Benutzen — gehört zum gemeinsamen Wissensbestand. Post-its tragen eine Beobachtung
von der Stelle, an der sie gemacht wurde, weiter zu anderen Werken.

Dieses Dokument hält durchgehend drei Stände auseinander:

| Kürzel | Bedeutung |
|---|---|
| **[A]** | Bestätigte Anforderung — vom Projektinhaber so vorgegeben |
| **[K]** | Konzept — Vorschlag dieser Fassung, nicht bestätigt, nicht gebaut |
| **[U]** | Umgesetzt — steht im Code und ist durch einen Test belegt |

---

## 1. Begriffe

| Begriff | Bedeutung | Stand |
|---|---|---|
| Mooncell / Mindcell | erschließt Themen und Positionen | [A] als Begriff · im Code nicht vorhanden |
| WERKSCHAU, FASSUNGSRAUM, GEGENÜBER | erschließen Verkörperungen und Varianten | [A] als Begriff · im Code nicht vorhanden |
| Post-it | verbindet beides: eine Erfahrung mit einem Werk kann eine Annahme verändern, und das bringt danach einen Text, eine Spielmechanik oder eine andere App weiter | [A] · Datenmodell und Weitergabe [U] |
| Werkepisode | sechs Bestandteile: Ausgangslage, Perspektiven, Bearbeitung, Erprobung, Entscheidung, Weiterwirkung | [A] · [U] |
| Entscheidung | übernommen, zurückgestellt, verworfen oder zurückgenommen | [A] · [U] |
| Rhythmus der Modelle | erst eigene Sicht entwickeln, dann Unterschiede öffnen, dann Alternativen erproben | [A] · nur [K], siehe 5.4 |
| „Was hat diese Erfahrung verändert?“ | von einer Beobachtung aus: betroffene Annahmen, Werkfassungen, Post-its, spätere Versuche, Veränderung des Anliegens | [A] · Endpunkt [U], Oberfläche offen |
| Invarianz der Identität („Was macht Zelda zu Zelda?“) | was ein Werk zu diesem Werk macht, zeigt sich oft erst beim Entwickeln und kann später revidiert werden | [A] · Ablageort [U], Verfahren offen |

Im Prototyp neu eingeführte Arbeitsbegriffe (alle **[K]**, Benennung frei änderbar):
**Werk**, **Werkfassung**, **Anliegen**, **Anliegen-Fassung**, **Annahme**,
**Verwendung** (eine Werkfassung nutzt eine Annahme an einer Stelle),
**Prüfauftrag** (eine Weiterwirkung, solange sie nicht geprüft ist), **Nachtrag**
(eine Episode, die eine abgeschlossene ergänzt).

## 2. Nachgeprüfte Befunde

Die Befunde der Vorsitzung wurden am Stand `2be374a` von `main` nachgeprüft.

1. **Bestätigt.** `markers` und `relations` hängen über `ON DELETE CASCADE` an
   `sessions`, `sparks` und `jobs`. Zusätzlich gefunden: `replace_markers` löscht
   bei jeder neuen Einschätzung alle Marker eines Funkens und legt sie neu an.
   Ein Marker ist also schon *innerhalb* seiner Sitzung nicht dauerhaft. Ein
   Post-it muss deshalb Zitat und Notiz mitnehmen und darf nicht nur verweisen.
2. **Mit Einschränkung bestätigt.** `context_snapshots` hält fest, was ein Auftrag
   zu sehen bekam. Zwei Einschränkungen: Die Tabelle hängt per CASCADE am Auftrag
   und verschwindet mit der Sitzung; und geschrieben wird mit `INSERT OR REPLACE`
   — die Unveränderlichkeit ist Vereinbarung, die Datenbank erzwingt sie nicht.
   Für die unabhängige Nachbearbeitung alter Episoden ist das **keine tragfähige
   Grundlage, solange Sitzungen gelöscht werden können** (siehe offene Frage 6.5).
3. **Bestätigt.** `relations.origin` / `relations.status` trennen maschinellen
   Vorschlag und bestätigten Befund. Die Weiterwirkung folgt dieser Regel.
4. **Bestätigt.** `tisch_shared_core/models.py` kennt die Quellenarten und die
   Kurationsstufen `raw … canonical`, aber weder Werkfassung noch Episode,
   Annahme oder Anliegen.
5. **Bestätigt.** `traumaatlas-2/src/kinformer/core/contract.js`: Der Auftrag
   nennt Werkstück, Fassung und verlangte Invarianten; das Ergebnis meldet
   `retained / changed / added / lost` und ist „erzeugt, nicht bestätigt“.
   Übernommen wurde die Denkfigur, kein Code.
6. **Gelesen, nicht angeschlossen.** `KIMI_HANDOFF_2026-10-09.md` samt Addendum.
7. **Bestätigt.** Die Begriffe aus Abschnitt 1 kamen in `der-tisch` bisher
   nicht vor. (Die anderen drei Repos wurden nicht erneut durchsucht.)

Eine weitere Beobachtung: XPERTENTiSCH hat **keinen Endpunkt, der eine Sitzung
löscht**. Der Test „Post-it überlebt das Löschen“ löscht deshalb direkt in der
Datenbank. Die Anforderung ist damit für den Fall belegt, dass es eine solche
Funktion einmal gibt oder jemand von Hand aufräumt.

## 3. Datenmodell

Alles **[U]**, soweit nicht anders vermerkt. Schema in
`xpertentisch/backend/app/db.py`, Endpunkte in `app/werkepisoden.py`.

```
anliegen ──< anliegen_fassungen
                    ▲                 ▲
                    │ (optional)      │ neue_anliegen_fassung_id
werke ──< werkfassungen               │
   │          ▲   ▲   ▲               │
   │          │   │   └── weiterwirkungen (Prüfauftrag: Ziel)
   │          │   └────── annahme_verwendungen >── annahmen
   │          └────────── postits >──< annahmen   (postit_annahmen, mit Haltung)
   │                         ▲
   └──< episoden ──< episoden_postits (Erprobung)
            └─────< entscheidungen ── neue_werkfassung_id
```

**Grundsatz:** Keine dieser Tabellen hat einen Fremdschlüssel auf `sessions`,
`sparks`, `jobs` oder `markers`. Die Herkunft eines Post-its ist ein Verweis
(`herkunft_art`, `herkunft_id`, `herkunft_sitzung`, `herkunft_job`), keine
Abhängigkeit. `markers` und `relations` sind unverändert.

| Tabelle | Inhalt | Regel |
|---|---|---|
| `anliegen`, `anliegen_fassungen` | Das Anliegen ist selbst versioniert | Eine Fassung wird nie überschrieben (Trigger) |
| `werke`, `werkfassungen` | Werk, Fassungsnummer, `zusammensetzung` (Liste von Stellen), optional Bezug auf eine Anliegen-Fassung, `invarianten`, `bilanz` | Eine Fassung wird nie überschrieben (Trigger) |
| `annahmen`, `annahme_verwendungen` | Eigene Kennung und Text; welche Werkfassung nutzt die Annahme an welcher Stelle | Eine Annahme gehört keinem Werk |
| `postits`, `postit_annahmen` | Verankert an Werkfassung und Stelle; Text, Zitat, Herkunft, `origin`, Status `offen / vorgeschlagen / bestaetigt / entkraeftet`; Haltung zur Annahme `einwand / stuetze / frage` | Was die Maschine anlegt, beginnt als `vorgeschlagen` |
| `episoden`, `episoden_postits`, `entscheidungen` | Die sechs Bestandteile, als Verweise | Abgeschlossen = unveränderlich (Endpunkt **und** Trigger) |
| `weiterwirkungen` | Prüfauftrag von Post-it oder Annahme zu einer anderen Werkfassung; `origin`, Status `vorgeschlagen / angenommen / abgelehnt / geprueft`; eigenes `ergebnis` (`betroffen / nicht_betroffen / unklar`) mit Notiz | Höchstens ein Auftrag je Quelle, Annahme und Ziel |

### 3.1 Die sechs Bestandteile einer Episode

| Bestandteil | Wo er liegt | Verweist auf |
|---|---|---|
| Ausgangslage | `episoden.ausgangslage` (Frage in eigenen Worten) | Werkfassung, Anliegen-Fassung, Aufträge (deren Kontext-Schnappschuss) |
| Perspektiven | `episoden.perspektiven` | Liste `{art, verweis, notiz}` — Modelle, Linsen, Annahmen, Erstbefunde |
| Bearbeitung | `episoden.bearbeitung` | Liste `{art, verweis, notiz}` |
| Erprobung | `episoden_postits` | Post-its als Beobachtung oder Gegenbeispiel |
| Entscheidung | `entscheidungen` | Gegenstand (Annahme, Post-it, Werkfassung, frühere Entscheidung), daraus hervorgegangene Werk- und Anliegen-Fassung |
| Weiterwirkung | abgeleitet | alle Prüfaufträge, die von den Post-its der Episode ausgehen |

Zwei Folgen dieser Bauweise, beide gewollt:

- **Die Weiterwirkung wächst nach dem Abschluss weiter.** Sie ist nicht Teil des
  eingefrorenen Inhalts, sondern wird beim Lesen gebildet. So ist die Episode
  abgeschlossen und das Werk trotzdem offen.
- **Eine Episode zeigt den heutigen Stand ihrer Post-its.** Wird eine Beobachtung
  später entkräftet, sieht man das auch in der alten Episode. Sie kopiert nichts.
  Was die Episode *damals* über das Post-it wusste, ist derzeit nicht festgehalten
  (offene Frage 6.6).

Eine frühere Entscheidung wird nicht geändert, sondern in einer späteren Episode
`zurueckgenommen` (Gegenstand: die frühere Entscheidung).

### 3.2 Weiterwirkung

Wird ein Post-it als **Einwand** auf eine Annahme bezogen, entsteht für jede
*andere* Werkfassung, die dieselbe Annahme nutzt, genau ein Prüfauftrag —
`origin: maschine`, Status `vorgeschlagen`, ohne Ergebnis. Der Schluss „gleiche
Annahme, also vielleicht auch betroffen“ ist maschinell und deshalb ein Vorschlag.
„Betroffen“ steht nur dort, wo ein Mensch ein Ergebnis eingetragen hat. Ein
eingetragenes Ergebnis wird nicht überschrieben; eine neue Erfahrung ist ein
neues Post-it. Stütze und Frage lösen keine Prüfaufträge aus.

Grenzen des Prototyps:
- „Andere Werkfassung“ schließt frühere Fassungen desselben Werks ein, sofern
  sie die Annahme nutzen. Ob das erwünscht ist, ist offen (6.7).
- Wird eine Annahme *nach* dem Einwand einer weiteren Werkfassung zugeordnet,
  entsteht kein Prüfauftrag nachträglich.

### 3.3 Invarianten

`werkfassungen.invarianten` ist eine Liste freier Sätze: was dieses Werk zu diesem
Werk machen soll, **nach dem Stand dieser Fassung**. Weil sie an der Fassung
hängt und nicht am Werk, kann die nächste Fassung sie anders fassen — die
Revidierbarkeit steckt in der Ablage. `werkfassungen.bilanz` hält nach dem
Vorbild des KiNFORMER-Vertrags fest, was gegenüber der Vorfassung `geblieben`,
`geaendert`, `hinzugekommen` und `verloren` ist.

Beides wird gespeichert und ausgegeben, **aber nicht geprüft**: Niemand gleicht
die Bilanz gegen die Invarianten ab, und es steht nirgends, wer eine Invariante
festgelegt hat (offene Frage 6.1).

## 4. Die Ansicht als Weg durch das Modell

`GET /api/postits/{id}/veraenderung` — **[U]** als Endpunkt, Oberfläche offen.

| Schritt | Frage | Weg im Modell | Feld der Antwort |
|---|---|---|---|
| 1 | Was wurde beobachtet, und wo? | `postits` → `werkfassungen` → `werke`, Anliegen-Fassung der Werkfassung, Herkunft | `postit`, `ausgangspunkt` |
| 2 | Welche Annahme trifft das? | `postit_annahmen` → `annahmen` | `annahmen[]` mit `haltung` |
| 3 | Wo wird dieselbe Annahme noch genutzt? | `annahme_verwendungen` → `werkfassungen`, je Fassung der Prüfauftrag aus `weiterwirkungen` | `annahmen[].werkfassungen[]` mit `stellen`, `pruefauftrag` |
| 4 | Wer hat das aufgenommen, was wurde entschieden? | `episoden_postits`, `entscheidungen` (zum Post-it oder zur Annahme), Episoden, in denen ein Prüfauftrag geprüft wurde | `episoden[].entscheidungen[]` |
| 5 | Was wurde danach versucht? | `entscheidungen.neue_werkfassung_id` | `spaetere_versuche[]` |
| 6 | Wie hat sich das Anliegen verändert? | `entscheidungen.neue_anliegen_fassung_id`, dazu die Fassung davor | `anliegen[]` mit `vorher`, `nachher` |

Dazu `luecken`: eine Liste in Klartext, wo der Weg endet — keine Annahme
zugeordnet, Prüfaufträge noch offen, nichts entschieden, Anliegen unverändert.
Die Ansicht soll nicht fertiger aussehen, als der Bestand ist.

„Spätere Versuche“ sind im Prototyp nur Fassungen, die eine Entscheidung
ausdrücklich nennt. Eine spätere Fassung, die niemand mit einer Entscheidung
verbunden hat, erscheint nicht.

## 5. Was umgesetzt ist — und was nicht

### 5.1 Umgesetzt und durch Tests belegt

`xpertentisch/backend/tests/test_werkepisoden.py`, 14 Tests; die 83 bisherigen
Tests laufen unverändert weiter (97 bestanden).

- Ein Post-it überlebt das Löschen seiner Herkunftssitzung, samt Zitat.
- Aus einem Marker wird ein Post-it; Marker und Beziehungen der Sitzung bleiben
  unverändert.
- Ein Einwand erzeugt je anderer Werkfassung genau einen Prüfauftrag im Stand
  `vorgeschlagen`; mehrere Stellen in einer Fassung und ein wiederholter Einwand
  erzeugen keinen zweiten.
- Jeder Prüfauftrag hat sein eigenes Ergebnis; ein Ergebnis wird nicht überschrieben.
- Was die Maschine anlegt (Post-it, Prüfauftrag), startet als Vorschlag.
- Eine abgeschlossene Episode weist Änderungen ab — an der Schnittstelle (409)
  und in der Datenbank selbst. Ein Nachtrag ist eine neue Episode.
- Die Ansicht liefert den Weg bis zur neuen Anliegen-Fassung und benennt Lücken.

### 5.2 Schnittstelle

```
POST /api/anliegen                         POST /api/anliegen/{id}/fassungen
GET  /api/anliegen/{id}
POST /api/werke        GET /api/werke      GET  /api/werke/{id}
POST /api/werke/{id}/fassungen             GET  /api/werkfassungen/{id}/postits
POST /api/annahmen     GET /api/annahmen/{id}
POST /api/annahmen/{id}/verwendungen
POST /api/postits      GET /api/postits/{id}
POST /api/postits/{id}/status              POST /api/postits/{id}/annahmen
POST /api/markers/{id}/postit
GET  /api/postits/{id}/veraenderung
POST /api/weiterwirkungen
POST /api/weiterwirkungen/{id}/status      POST /api/weiterwirkungen/{id}/ergebnis
POST /api/episoden     GET /api/episoden/{id}    POST /api/episoden/{id}
POST /api/episoden/{id}/postits            POST /api/episoden/{id}/entscheidungen
POST /api/episoden/{id}/abschluss
```

### 5.3 Bewusst nicht umgesetzt

- Keine Oberfläche. Die Leseansicht folgt erst nach Rückfrage.
- Kein Zugangsschutz für diese Endpunkte (wie die übrigen Sitzungs-Endpunkte).
- Keine Ereignisse im Ereignisstrom: der ist an Sitzungen gebunden.
- Kein Anschluss an `tisch_shared_core`, an KiNTEGRiTY TRAGWERK oder an KiNFORMER.
- Mooncell/Mindcell, WERKSCHAU, FASSUNGSRAUM und GEGENÜBER sind nicht modelliert.
  Das Modell bietet Anschlussstellen (Annahmen ↔ Themen und Positionen;
  Werkfassungen ↔ Verkörperungen und Varianten), mehr nicht.
- Kein Löschen und kein Bearbeiten von Post-it-Texten, Annahmen oder Werken.

### 5.4 Konzept, nicht gebaut

- **Rhythmus der Modelle [K].** `perspektiven` ist eine ungeordnete Liste. Der
  Dreischritt — eigene Sicht, Unterschiede öffnen, Alternativen erproben — ließe
  sich als Phase je Verweis führen und an der Schnittstelle erzwingen: Ein
  Erstbefund zählt nur als unabhängig, wenn der Kontext-Schnappschuss seines
  Auftrags keinen anderen Erstbefund enthält. Das ist prüfbar, weil der
  Schnappschuss die gesehenen Beiträge aufzählt.
- **Unabhängige Nachbearbeitung alter Episoden [K].** Ein Endpunkt, der einem
  neuen Modell nur Ausgangslage und Perspektiven einer Episode ausgibt — ohne
  Erprobung und Entscheidung. Voraussetzung ist Frage 6.5.
- **Invarianten prüfen [K].** Abgleich der `bilanz` einer neuen Fassung gegen die
  `invarianten` der vorigen: Was als verloren gemeldet wird und zugleich
  Invariante war, ist ein Anlass für ein Post-it, kein Fehler.

## 6. Offene Fragen

Die ersten drei sind ausdrücklich **nicht** in dieser Fassung entschieden.

1. **Wer legt Invarianten fest?** Der Prototyp speichert sie ohne Urheber und ohne
   Tor. Denkbar: nur der Mensch; Maschine schlägt vor (wie bei Post-its); oder
   erst rückblickend aus mehreren Fassungen.
2. **Gehören Episoden in den `tisch_shared_core` oder bleiben sie in XPERTENTiSCH?**
   Der Prototyp liegt in XPERTENTiSCH. Für den Core spräche, dass Werke und
   Annahmen appübergreifend sind; dagegen, dass der Core bisher Quellen und
   Kurationsstufen kennt und keine Verläufe.
3. **Wird die hashverkettete Akte von KiNTEGRiTY angeschlossen?** Sie würde die
   Unveränderlichkeit abgeschlossener Episoden nachweisbar statt nur erzwungen
   machen. Das Angebot liegt vor; angeschlossen ist nichts.
4. **Wer darf einen Prüfauftrag annehmen oder ein Ergebnis eintragen?** Derzeit
   jeder, der die Schnittstelle erreicht. Es gibt keine Personen im Modell.
5. **Was geschieht mit Kontext-Schnappschüssen, wenn eine Sitzung gelöscht wird?**
   Heute verschwinden sie. Soll eine Episode ihre Ausgangslage festhalten können,
   muss der Schnappschuss entweder die Sitzung überleben oder beim Anlegen der
   Episode übernommen werden — das wäre die eine Stelle, an der kopiert wird.
6. **Soll eine abgeschlossene Episode den damaligen Stand ihrer Post-its
   festhalten?** Heute zeigt sie den heutigen.
7. **Wandert ein Einwand auch zu früheren Fassungen desselben Werks?** Heute ja.
8. **Wie fein ist eine „Stelle“?** Heute ein freier Text, der zur
   `zusammensetzung` passen kann, aber nicht muss.
9. **Darf die Maschine Annahmen aus Werken herauslesen und vorschlagen?** Annahmen
   haben im Prototyp weder `origin` noch Status.
10. **Was ist ein „späterer Versuch“, wenn niemand ihn mit einer Entscheidung
    verbunden hat?**

## 7. Außerhalb dieses Auftrags

Das Prüfpaket ATLAS-2026-10-09-R1 (`00_START.md`, `02_Pruefprompt_Alle_LLMs.txt`,
die sieben Atlanten) wurde bisher nirgends gefunden. Diese Prüfung gehört nicht
zu diesem Auftrag und wurde nicht begonnen.
