# ANALYSE B — Kampfmittelsondierung im HDD-Verfahren für 6 Rohrvortriebe DN 2800

**Status:** Entscheidungsvorlage, keine Entscheidung. Stand 17.09.2026.
**Grundlage:** weitergeleitete Mail (Wortlaut siehe Ausgangslage) und LV-Text „Technische Bearbeitung". **Die Skizze und die Zeichnungen lagen bei der Erstellung nicht vor.** Alle Geometrie- und Längenangaben in diesem Dokument sind deshalb Parameter, keine Projektwerte.

## Lesehilfe: Kennzeichnung der Aussagen

| Kürzel | Bedeutung |
|---|---|
| **[Q]** | Quelle: steht so in der Mail oder im LV-Text. |
| **[H]** | Herleitung: folgt rechnerisch oder fachlich aus [Q] und den genannten Annahmen. |
| **[A]** | Annahme: von mir gesetzt, weil die Information fehlt. Muss vor Verwendung bestätigt oder ersetzt werden. |
| **[N]** | Nachzuschlagen: die Vorgabe existiert, die konkrete Zahl/Fassung ist im genannten Regelwerk zu prüfen. Ich erfinde sie nicht. |

## Ausgangslage (Zusammenfassung der Quelle)

- [Q] RTM ist als Nachunternehmer vorgesehen. Leistung: Kampfmittelsondierung mittels 3-Achs-Sondierung im HDD-Verfahren für Rohrleitungs-Vortriebe DN 2800.
- [Q] 6 Vortriebe an 2 Kreuzungen. Baubeginn 05.10.2026.
- [Q] Strittig: 3 HDD-Bohrungen je Vortrieb an einer Stelle, 7 Bohrungen je Vortrieb an anderer Stelle. Eiffage klärt mit dem Auftraggeber.
- [Q] Preise: 45,00 €/m HDD-Bohrung, 2,00 €/m Rohr DN 70.
- [Q] Abstimmung mit Rai in der kommenden Woche.
- [Q] LV-Text „Technische Bearbeitung" (Wortlaut siehe Abschnitt 2).
- [H] 6 Vortriebe an 2 Kreuzungen ergibt 3 Vortriebe je Kreuzung. Das ist die erste mögliche Quelle der Verwechslung „3": siehe Abschnitt 1.6 und 8.
- [H] Lagebezug EPSG 31466 (DHDN / Gauß-Krüger Zone 2, Bezugsmeridian 6° Ost) deutet auf ein Projektgebiet im Westen Deutschlands (Streifen um 6° Ost, z. B. Rheinland, westliches NRW, RLP, Saarland). Welches Bundesland und damit welcher Kampfmittelbeseitigungsdienst (KBD) zuständig ist, folgt daraus **nicht** eindeutig. [N] Zuständigen KBD über den Auftraggeber feststellen.
- [A] Der Leser dieses Dokuments („eigene Firma") ist Vertragspartner von Eiffage und bindet RTM als Nachunternehmer für den HDD-Teil ein. Ob die kampfmitteltechnische Fachleistung (Sondierung, Auswertung, Freigabe) bei der eigenen Firma, bei RTM oder bei einem Dritten liegt, ist aus der Mail nicht ersichtlich. Genau das ist der wichtigste Klärungspunkt (Abschnitt 3, Punkt 1).

---

## 1. Fachliche Prüfung: 3 oder 7 Bohrungen je Vortrieb

### 1.1 Prinzip

Jede HDD-Bohrung ist eine **Sondierachse**. Nach dem Pilotbohren und dem Ziehen der Stahl-Bohrgestänge wird ein nichtmagnetisches Rohr (vermutlich das genannte Rohr DN 70, [A]) eingezogen, durch das die 3-Achs-Magnetometersonde geschoben oder gezogen wird. Die Sonde erfasst ferromagnetische Störkörper bis zu einem **anrechenbaren Sondierradius r_d** um die Sondierachse. Im Querschnitt ist die Abdeckung einer Sondierachse damit ein Kreis mit Radius r_d, entlang der Bohrung ein Zylinder.

Die Anzahl der Bohrungen ergibt sich aus der Frage: **Wie viele Kreise mit Radius r_d braucht es, um den freizugebenden Querschnitt lückenlos abzudecken?**

### 1.2 Was freizugeben ist: Freigaberadius R_F

| Größe | Wert | Kennzeichnung |
|---|---|---|
| Nennweite Vortrieb | DN 2800 | [Q] |
| Außendurchmesser Vortriebsrohr D_a | **3,50 m** | [A] Stahlbeton-Vortriebsrohr DN 2800 mit Wandstärke ~ 0,30–0,40 m; tatsächlichen Wert aus der Rohrstatik/Zeichnung übernehmen. Der Überschnitt der Vortriebsmaschine kommt hinzu. |
| Außenradius R_a = D_a / 2 | 1,75 m | [H] |
| Arbeitsraum / Sicherheitszuschlag s um das Rohr | **0,5 / 1,0 / 1,5 m** (parametrisch) | [A] Wie groß der freizugebende Raum um den Vortriebsquerschnitt sein muss (Ringspalt, Schmierung, Auflockerungszone, Sicherheitsabstand zum Zündmechanismus), legt der zuständige KBD in Abstimmung mit dem Auftraggeber fest. [N] Vorgabe des KBD / BFR KMR, Abschnitt zu Räumzielen und Freigabevolumen. |
| Freigaberadius R_F = R_a + s | **2,25 / 2,75 / 3,25 m** | [H] |

### 1.3 Was eine Sondierachse leistet: Sondierradius r_d und wirksamer Radius r_eff

- Der **Sondierradius r_d** ist keine Gerätekonstante. Er hängt ab vom kleinsten nachzuweisenden Objekt (Zielobjektklasse: Sprengbombe 250 kg / 50 kg / Granaten), von der Sondenempfindlichkeit, vom Störpegel (Spundwände, Gleise, Leitungen, Bewehrung) und von dem, was der KBD anerkennt. [N] Fundstellen: Vorgaben/Merkblatt des zuständigen KBD zur Bohrlochsondierung (dort sind Rasterabstände bzw. anrechenbare Reichweiten je Zielobjekt hinterlegt); BFR KMR, Anhang zu Sondierverfahren / Tiefensondierung (Anhangsnummer in der aktuellen Ausgabe prüfen); Herstellerangaben der eingesetzten Sonde. **Ich setze hier keinen Wert als Fakt.**
- Für die Rechnung wird r_d parametrisch mit **1,0 / 1,5 / 2,0 / 2,5 m** angesetzt. [A]
- **Lagetoleranz t der HDD-Bohrung:** Die Bohrung liegt nie exakt auf der Planachse. Der wirksame Radius ist r_eff = r_d − t. [A] t = 0,3 m bei Bohrlochvermessung (Wireline/Gyro), t = 0,5 m bei reiner Walkover-Ortung. Unter Gewässern, Gleisen oder tiefen Straßen ist Walkover meist nicht möglich. Maßgeblich ist die **eingemessene Ist-Lage**, nicht die Planlage (siehe 1.7).

### 1.4 Deckungsgeometrie (Rechnung)

Optimale Überdeckung eines Kreises mit Radius R_F durch n gleiche Kreise mit Radius r_eff (bekannte Ergebnisse der Kreisüberdeckung, [H]):

| n | Anordnung | Bedingung | Lage der Bohrachsen |
|---|---|---|---|
| 1 | zentrisch | r_eff ≥ R_F | auf der Vortriebsachse |
| 3 | 120°-Dreieck | r_eff ≥ 0,866 · R_F | Abstand 0,5 · R_F von der Vortriebsachse |
| 7 | 1 zentrisch + 6 im Sechseck | r_eff ≥ 0,5 · R_F | Zentrum auf der Achse, 6 Bohrungen im Abstand 0,866 · R_F |

Das sind Mindestbedingungen bei **tangierender** Überdeckung. Die KBD verlangen üblicherweise eine **Überlappung** [N], das heißt in der Praxis ist r_eff spürbar größer zu wählen, als die Tabelle sagt.

**Erforderlicher nominaler Sondierradius r_d = Bedingung + t (mit t = 0,3 m [A]):**

| Freigaberadius R_F | 3 Bohrungen | 7 Bohrungen |
|---|---|---|
| 2,25 m (s = 0,5 m) | r_d ≥ 2,25 m | r_d ≥ 1,43 m |
| 2,75 m (s = 1,0 m) | r_d ≥ 2,68 m | r_d ≥ 1,68 m |
| 3,25 m (s = 1,5 m) | r_d ≥ 3,11 m | r_d ≥ 1,93 m |

**Umgekehrt: welcher Freigaberadius wird bei gegebenem r_d abgedeckt (t = 0,3 m [A]):**

| r_d nominal | r_eff | 1 Bohrung | 3 Bohrungen | 7 Bohrungen |
|---|---|---|---|---|
| 1,0 m | 0,7 m | 0,70 m | 0,81 m | 1,40 m |
| 1,5 m | 1,2 m | 1,20 m | 1,39 m | 2,40 m |
| 2,0 m | 1,7 m | 1,70 m | 1,96 m | 3,40 m |
| 2,5 m | 2,2 m | 2,20 m | 2,54 m | 4,40 m |

Zum Vergleich: allein das Rohr (ohne jeden Zuschlag) hat R_a = 1,75 m.

### 1.5 Was die Rechnung zeigt [H]

1. **3 Bohrungen decken den Rohrquerschnitt DN 2800 plus Arbeitsraum nur, wenn der KBD einen anrechenbaren Sondierradius von rund 2,3 bis 3,1 m anerkennt.** Ob das der Fall ist, hängt an der Zielobjektklasse. Bei r_d = 1,5 m decken 3 Bohrungen nicht einmal das Rohr selbst (1,39 m < 1,75 m). Bei r_d = 2,0 m decken sie das Rohr knapp und keinen Arbeitsraum.
2. **7 Bohrungen decken bei r_d ≥ ~1,7 m den Querschnitt mit 1,0 m Arbeitsraum**, bei r_d = 2,0 m sogar mit ~1,6 m Zuschlag. Die 7er-Anordnung ist die klassische Sechseck-Überdeckung und entspricht dem, was man von einem Fachplaner für DN 2800 erwarten würde.
3. Bei der 3er-Anordnung liegen die Bohrachsen **innerhalb** des Rohrquerschnitts (0,5 · R_F ≈ 1,1–1,6 m von der Achse); alle drei verdämmten Bohrkanäle und Rohre DN 70 werden später von der Vortriebsmaschine durchfahren. Bei der 7er-Anordnung liegt nur die Zentralbohrung im Querschnitt, die 6 äußeren liegen bei 0,866 · R_F ≈ 2,0–2,8 m knapp außerhalb der Rohrwand. Das ist für die Verdämmung und für die Maschine relevant (Abschnitt 5 und 6).
4. Die tatsächliche Anordnung auf der Skizze kann von den Optimalanordnungen abweichen (z. B. 6 Bohrungen auf einem Ring an der Rohrwand plus Zentrum). Dann ist die Deckung **mit der Ist-Geometrie der Skizze neu zu rechnen**; die Formeln oben sind der Maßstab, nicht das Ergebnis.

### 1.6 Andere Lesarten der „3" [H]

- 6 Vortriebe an 2 Kreuzungen = **3 Vortriebe je Kreuzung**. Es ist gut möglich, dass „3 je …" an einer Stelle Vortriebe je Kreuzung meint, an der anderen Stelle „7 Bohrungen je Vortrieb" Sondierachsen. Das wäre kein fachlicher Widerspruch, sondern ein Lesefehler. Vor jeder Diskussion über Sondierradien: **Beide Textstellen im Original nebeneinanderlegen.**
- „3-Achs-Sondierung" bezeichnet die dreiachsige Magnetometersonde (drei Messachsen x/y/z). Auch das kann zu „3" verkürzt worden sein.
- Denkbar ist auch, dass die 3 Bohrungen aus einem älteren Konzept für einen kleineren Vortriebsdurchmesser stammen.

### 1.7 Ergebnis der Prüfung

**Fachlich trägt die 7**, unter folgender Bedingung: Der vom zuständigen KBD anerkannte Sondierradius, abzüglich der nachgewiesenen Lagetoleranz der Bohrung, ist mindestens die Hälfte des Freigaberadius (r_d − t ≥ 0,5 · R_F), und die KBD-Vorgabe zur Überlappung ist eingehalten. Mit den hier angenommenen Werten (D_a = 3,5 m, s = 1,0 m, t = 0,3 m) heißt das r_d ≥ ~1,7 m nominal.

**Die 3 trägt nur**, wenn der KBD für die maßgebliche Zielobjektklasse einen Sondierradius von rund 2,3 bis 3,1 m schriftlich anerkennt **oder** das Freigabeziel kleiner ist als „Rohrquerschnitt plus Arbeitsraum". Beides ist nicht belegt. Ohne schriftliche KBD-Bestätigung ist die 3 nicht kalkulier- und nicht verantwortbar.

Empfehlung für die Angebotsphase: **Kalkulation auf 7 als Hauptposition, 3 als Alternativposition mit dem Vorbehalt „nur bei schriftlicher KBD-Anerkennung des Sondierradius ≥ x m".** Die Freigabe gilt stets nur für das Volumen, das mit der **eingemessenen Ist-Lage** der Bohrungen tatsächlich überdeckt wurde; weicht eine Bohrung ab, entsteht eine Lücke, die nachzubohren ist (Risikoposition, Abschnitt 7).

---

## 2. Gliederung der technischen Bearbeitung nach LV-Text

LV-Wortlaut [Q]: *Technische Bearbeitung der Bohrungen unter Beachtung von DIN 18324, DWA A 125, der Richtlinien der DCA sowie DVGW GW 321, GW 301 und GW 302. Vorhalten und Betrieb der erforderlichen messtechnischen Einrichtungen. Festlegung von Trasse und Gradiente. Lagebezug DHDN / 3-Grad Gauß-Krüger Zone 2 (EPSG 31466), Höhenbezug DHHN92 (EPSG 5783). Planung und Ausführung der Anfahr- und Einfahrphasen. Standsicherheitsnachweise. Projektplan. Verdämmen des Bohrkanals. Dokumentation der Sondierergebnisse. Ggf. Einholen der Genehmigung für Nacht-, Sonn- und Feiertagsarbeit.*

Zuordnung [H], Spalten „RTM" = HDD-Nachunternehmer, „Eigen" = eigener Verantwortungsbereich (Annahme aus der Ausgangslage), „Offen" = weder zugeordnet noch belegt:

| LV-Position | Inhalt (fachlich) | RTM liefert | Eigen | Offen / Bemerkung |
|---|---|---|---|---|
| Regelwerke DIN 18324, DCA, GW 321 | HDD-Ausführung: VOB/C Horizontalspülbohrarbeiten, DCA-Technische Richtlinien, DVGW-Arbeitsblatt für steuerbare Spülbohrverfahren | Ausführung, Bohrprotokolle, Spülungsmanagement | Prüfung der Nachweise | — |
| GW 301 / GW 302 | Qualifikations- und Zertifizierungsnachweise des Rohrleitungsbauunternehmens (GW 301) bzw. für grabenlose Verfahren (GW 302) | Zertifikate von RTM in der einschlägigen Verfahrensgruppe | Weitergabe an Eiffage/AG | [N] Ob die Zertifizierung für die konkrete HDD-Klasse vorliegt: bei RTM anfordern. |
| DWA-A 125 | Regelwerk **Rohrvortrieb**, nicht HDD | — | — | **Widerspruch:** DWA-A 125 betrifft den Vortrieb DN 2800, nicht die Sondierbohrungen. Vermutlich aus dem Vortriebs-LV übernommen. Klären, was davon RTM treffen soll (Abschnitt 8). |
| Messtechnische Einrichtungen, Vorhalten und Betrieb | (a) HDD-Ortung/Bohrlochvermessung (Walkover, Wireline, Gyro) (b) 3-Achs-Magnetometersonde mit Datenlogger und Wegmessung (c) Vermessung (Absteckung, Einmessung Ein-/Austrittspunkte) | (a) sicher, (c) in der Regel | (b) je nach Rollenverteilung | **Offen:** Wer stellt und betreibt die Sondiersonde? Das ist eine Kampfmittel-Fachleistung (Abschnitt 5.3). |
| Festlegung von Trasse und Gradiente | Lage jeder Sondierachse relativ zur Vortriebsachse (Abschnitt 1), Eintritts-/Austrittspunkte, Bogenradien nach Gestänge, Tiefe der Parallelstrecke | Bohrtechnische Trassierung (Radien, Ein-/Austrittswinkel) | Sondiergeometrie (Abstände, Deckung, Abstimmung KBD) | Die Sondiergeometrie darf nicht der Bohrfirma allein überlassen werden. Sie ist Teil der Freigabelogik. |
| Lagebezug EPSG 31466 / Höhenbezug EPSG 5783 | Alle Daten (Bohrachsen Ist-Lage, Sondierdaten, Anomalien, Freigabekorridor) in diesem System | Ist-Lage der Bohrungen | Sondier- und Freigabedaten im selben System | **Hinweis [H]:** DHHN92 ist in den meisten Ländern durch DHHN2016 (EPSG 7837) abgelöst, DHDN/GK durch ETRS89/UTM. Unterschiede im cm- bis dm-Bereich, regional verschieden. Wer die Transformation macht und mit welcher Methode (NTv2-Gitter des Landes), ist festzulegen; sonst passen Sondierdaten und Vortriebsachse später nicht zusammen. |
| Anfahr- und Einfahrphasen | **Mehrdeutig.** HDD-Lesart: Eintritt (Anfahren) von der Oberfläche in die Parallelstrecke und Austritt. Vortriebs-Lesart: Anfahren der Vortriebsmaschine aus dem Startschacht durch die Anfahrdichtung, Einfahren in den Zielschacht | HDD-Lesart: ja | — | **Widerspruch:** Die Begriffe stammen aus der Vortriebstechnik. Wenn Vortriebs-Anfahren gemeint ist, gehört das nicht in ein Sondier-LV. Klären (Abschnitt 8). |
| Standsicherheitsnachweise | HDD-Lesart: Nachweis Spüldruck gegen Überlagerung (Ausbläser-/Frac-out-Nachweis), Bohrlochstabilität, ggf. Nachweis für Kreuzungsbauwerk (Straße, Gleis, Gewässerdamm) nach Vorgabe des Kreuzungsträgers | Bohrtechnische Nachweise | Prüfung, Weiterleitung an Kreuzungsträger | Nachweise für den Vortrieb DN 2800 selbst gehören **nicht** hierher. Vom Kreuzungsträger (Straßenbaulastträger, DB, Wasserstraßenverwaltung) geforderte Nachweisformate sind [N]. |
| Projektplan | Ablauf- und Terminplan der Sondierbohrungen je Kreuzung, mit Schnittstellen zu Schachtbau, KBD-Freigabe, Vortriebsbeginn | Bohrtechnischer Terminplan | Gesamtplan inkl. Freigabelauf | Kritisch wegen Baubeginn 05.10.2026 (Abschnitt 3, Punkt 3). |
| Verdämmen des Bohrkanals | Verfüllen jedes Bohrkanals und des Rohrs DN 70 nach der Sondierung mit Dämmer (z. B. Bentonit-Zement), mit Mengennachweis | Ausführung | Abnahme der Verfüllprotokolle | Ob das Rohr DN 70 verbleibt und verfüllt wird oder gezogen wird: klären (Abschnitt 6). |
| Dokumentation der Sondierergebnisse | Sondierbericht: Rohdaten, Ist-Lage der Achsen, Anomalienliste mit Koordinaten, Freigabekorridor, Freigabebescheinigung | Ist-Lage der Bohrungen, Bohrprotokolle | Sondierbericht und Freigabe (falls die eigene Firma die Fachfirma ist) | **Offen:** Wer unterschreibt die Freigabe? (Abschnitt 5.3). Format nach KBD-Vorgabe [N]. |
| Genehmigung Nacht-, Sonn-, Feiertagsarbeit | Antrag bei der zuständigen Behörde (Gewerbeaufsicht/Ordnungsamt), Lärmschutz | Antrag für eigene Arbeiten | Koordination | „Ggf." heißt: nur wenn der Terminplan es erzwingt. Bei Baubeginn 05.10.2026 ist das wahrscheinlich (Abschnitt 7). |

**Nicht im LV-Text, aber für Kampfmittelsondierung zwingend [H]:** Sondierung der Bohransatzpunkte vor dem Bohren, Gefährdungsbeurteilung Kampfmittel, Abstimmung mit dem KBD, Verantwortliche Person nach Sprengstoffgesetz, Sicherheits- und Notfallplan bei Fund. Das LV liest sich wie ein reines HDD-Leistungsverzeichnis, dem der Kampfmittel-Teil fehlt (Abschnitt 8).

---

## 3. Offene Punkte für die Abstimmung mit Rai

Jeder Punkt mit der Konsequenz, wenn er offen bleibt.

| Nr. | Offener Punkt | Konsequenz, wenn offen |
|---|---|---|
| 1 | **Wer ist die Kampfmittel-Fachfirma?** Wer führt die Sondierung durch, wertet aus, unterschreibt die Freigabebescheinigung, stellt die Verantwortliche Person nach SprengG? RTM (Bohrfirma), die eigene Firma oder ein Dritter? | Ohne Fachfirma gibt es keine Freigabe. Ohne Freigabe darf der Vortrieb nicht starten. Haftung und Versicherung sind nicht zuordenbar. **Das ist der Punkt, an dem alles andere hängt.** |
| 2 | **3 oder 7 Bohrungen** (Abschnitt 1), inklusive Prüfung, ob „3" die Vortriebe je Kreuzung meint. | Mengengerüst schwankt um den Faktor 2,3 (Abschnitt 4). Ein Angebot auf 3 ohne KBD-Bestätigung ist fachlich nicht haltbar. |
| 3 | **Was bedeutet „Baubeginn 05.10.2026"?** Beginn der Sondierbohrungen, Beginn des Schachtbaus oder Beginn des Vortriebs? Heute ist der 17.09.2026, also 18 Kalendertage. | Wenn der Vortrieb am 05.10. beginnt, müssen bis dahin Sondierung, Auswertung und Freigabe mindestens für den ersten Vortrieb abgeschlossen sein. Das ist mit KBD-Abstimmung, Genehmigungen und bis zu 42 Bohrungen nicht realistisch. Dann steht Stillstand des Vortriebs im Raum, und die Frage, wer ihn bezahlt. |
| 4 | **Zuständiger KBD und dessen Vorgaben:** Zielobjektklasse, anerkannter Sondierradius, Überlappung, Freigabevolumen (s), akzeptierte Sondentechnik, Berichtsformat. | Ohne diese Zahlen ist die Deckungsrechnung nicht abschließbar und die Bohrungsanzahl nicht belastbar. Nachträgliche Verschärfung durch den KBD bedeutet Nachbohrungen. |
| 5 | **Bohrlänge je Bohrung und Bohrkonzept:** Länge der Vortriebe, Tiefe der Achse, Start von der Oberfläche (mit Ein-/Austrittsbögen) oder aus dem Schacht. | Der Metrepreis ist ohne Länge keine Summe. Oberflächenstart verlängert jede Bohrung um die Ein- und Austrittsstrecke (Abschnitt 4.2). Schachtstart setzt Schachtgröße und Bauzeitfolge voraus. |
| 6 | **Was ist im Preis 45 €/m enthalten?** Und: ist das RTM-Angebot oder Eiffage-Zielpreis? Pilotbohrung, Aufweitung, Rohreinzug, Spülung, Verdämmung, Vermessung, BE, Mobilisierung je Kreuzung? | Jede nicht enthaltene Position ist entweder Nachtrag oder eigener Verlust (Abschnitt 7). |
| 7 | **Rohr DN 70:** Material (muss nichtmagnetisch sein: PE/PVC/GFK, keine Stahlmuffen), Innendurchmesser zur Sonde passend, Verbleib (verfüllt im Boden oder gezogen), Verhalten beim Durchfahren durch die Vortriebsmaschine. „DN 70" ist keine übliche PE-Nennweite; gemeint ist vermutlich d 75. | Falsches Rohr = Sondierung nicht durchführbar oder Messdaten gestört. Rohrreste im Ortsbrustbereich können sich in Abbauwerkzeugen verfangen. |
| 8 | **Kreuzungsträger:** Was wird gekreuzt (Straße, Bahn, Gewässer, Damm)? Welche Auflagen (Bahn: Betra/Sperrpausen, Gewässer: wasserrechtliche Erlaubnis, Straße: verkehrsrechtliche Anordnung)? | Genehmigungslaufzeiten von Wochen bis Monaten. Unter Gleisen und Gewässern ist Walkover-Ortung nicht möglich; die Bohrlochvermessung muss anders gelöst werden (Kosten, Toleranz t). |
| 9 | **Freigabe des Bohransatzpunktes und Bohren in nicht freigegebenem Boden:** Akzeptiert der KBD das Pilotbohren durch unsondierten Boden, und unter welchen Auflagen? | Wenn nein: Sondierbegleitetes Bohren oder Vorab-Sondierung der Bohrtrasse, das heißt ein anderes Verfahren und andere Kosten. Wenn ungeklärt: Personengefährdung beim Bohren (Abschnitt 5.2). |
| 10 | **Lage der Bohrungen relativ zum Vortrieb:** Skizze prüfen. Können benachbarte Vortriebe (3 je Kreuzung) Bohrungen teilen? | Mögliche Einsparung gegenüber 3 × 7 = 21 Bohrungen je Kreuzung. Ohne Skizze nicht rechenbar. |
| 11 | **Reihenfolge:** Alle Sondierbohrungen einer Kreuzung vor dem ersten Vortrieb dort? Stahlbewehrung/Stahlrohre eines fertigen Vortriebs stören die Sondierung des Nachbarn. | Bei falscher Reihenfolge sind Teilbereiche nicht auswertbar und damit nicht freigebbar. |
| 12 | **Koordinatensystem:** Wer transformiert von/zu ETRS89/UTM bzw. DHHN2016, mit welcher Methode? | Lagefehler im dm-Bereich zwischen Freigabekorridor und Vortriebsachse. Bei r_eff-Reserven von wenigen Dezimetern ist das nicht tolerierbar. |
| 13 | **Fund-Szenario vertraglich:** Wer trägt Stillstand, Umplanung, Evakuierung, Bergung? | Ohne Regelung liegt das Risiko beim Auftragnehmer, der es am wenigsten steuern kann. |
| 14 | **Versicherung:** Deckt die Haftpflicht von RTM und der eigenen Firma Kampfmittelrisiken? Diese sind in Standardpolicen häufig ausgeschlossen [N]. | Im Schadensfall keine Deckung. |

---

## 4. Kosten- und Mengengerüst

### 4.1 Grundwerte

| Größe | Wert | Kennzeichnung |
|---|---|---|
| Vortriebe | 6 | [Q] |
| Bohrungen je Vortrieb | 3 bzw. 7 | [Q] |
| Bohrungen gesamt | 18 bzw. 42 | [H] |
| Einheitspreis HDD-Bohrung | 45,00 €/m | [Q] |
| Einheitspreis Rohr DN 70 | 2,00 €/m | [Q] |
| Summe je Meter (Rohr über volle Bohrlänge) | 47,00 €/m | [H], [A] Rohr über volle Länge |
| Bohrlänge je Bohrung L | **50 / 100 / 150 / 200 m** | [A] Die Vortriebslängen sind nicht bekannt. Die Spanne deckt kurze Straßenkreuzungen bis längere Gewässer- oder Bahnkreuzungen ab. **Muss durch die Ist-Längen aus der Skizze ersetzt werden.** |

### 4.2 Was „Bohrlänge" bedeutet [H]

Bei Start von der Oberfläche ist jede HDD-Bohrung länger als der Vortrieb: L_Bohrung ≈ L_Vortrieb + L_Eintritt + L_Austritt + Überlappung an den Schächten. Die Ein- und Austrittsstrecken hängen von Achstiefe h und Eintrittswinkel α ab (grob L_Eintritt ≈ h / tan α zuzüglich Bogen; [A] α = 10–15°, Gestängeradius nach Herstellerangabe). Bei h = 8 m und α = 12° sind das rund 40 m je Seite, also bis zu 80 m Mehrlänge je Bohrung, die nach Metrepreis bezahlt werden, aber keine Sondierleistung für den Vortrieb erbringen. Bei Start aus dem Schacht entfällt das, dafür braucht es Schachtgröße, Bauzeitfolge und ein kleines Bohrgerät.

**Ob der Metrepreis für die gesamte Bohrlänge oder nur für die Sondierstrecke gilt, ist zu klären (Abschnitt 3, Punkt 6).**

### 4.3 Mengen und Summen (nur Einheitspreise, ohne Nebenkosten)

**Variante 3 Bohrungen je Vortrieb (18 Bohrungen):**

| L je Bohrung [A] | Bohrmeter gesamt | HDD 45 €/m | Rohr 2 €/m | Summe |
|---|---|---|---|---|
| 50 m | 900 m | 40.500 € | 1.800 € | **42.300 €** |
| 100 m | 1.800 m | 81.000 € | 3.600 € | **84.600 €** |
| 150 m | 2.700 m | 121.500 € | 5.400 € | **126.900 €** |
| 200 m | 3.600 m | 162.000 € | 7.200 € | **169.200 €** |

**Variante 7 Bohrungen je Vortrieb (42 Bohrungen):**

| L je Bohrung [A] | Bohrmeter gesamt | HDD 45 €/m | Rohr 2 €/m | Summe |
|---|---|---|---|---|
| 50 m | 2.100 m | 94.500 € | 4.200 € | **98.700 €** |
| 100 m | 4.200 m | 189.000 € | 8.400 € | **197.400 €** |
| 150 m | 6.300 m | 283.500 € | 12.600 € | **296.100 €** |
| 200 m | 8.400 m | 378.000 € | 16.800 € | **394.800 €** |

Faktor zwischen den Varianten: 7/3 = 2,33. Die Differenz liegt je nach Länge zwischen rund 56.000 € und 226.000 € (nur Einheitspreise).

### 4.4 Zeitgerüst [A]

Annahme: eine Bohrung von 100 m inklusive Einrichten, Pilotbohrung, Rohreinzug, Sondierung, Verdämmen, Umsetzen = **1 bis 2 Arbeitstage je Bohrgerät**. Diese Zahl ist eine Setzung zur Größenordnung und durch RTMs Leistungsansatz zu ersetzen.

| Variante | Bohrungen | Arbeitstage bei 1 Gerät | bei 2 Geräten |
|---|---|---|---|
| 3 je Vortrieb | 18 | 18–36 | 9–18 |
| 7 je Vortrieb | 42 | 42–84 | 21–42 |

Hinzu kommen Auswertung, Berichtserstellung und Freigabelauf beim KBD, deren Dauer [N] beim zuständigen KBD zu erfragen ist. Gegen den 05.10.2026 gerechnet ist die 7er-Variante nur dann haltbar, wenn „Baubeginn" nicht „Vortriebsbeginn" heißt oder die erste Kreuzung mit hoher Priorität und mehreren Geräten gefahren wird.

### 4.5 Sensitivität

Der größte Hebel ist nicht der Einheitspreis, sondern (1) die Bohrungsanzahl, (2) die Bohrlänge inklusive Ein-/Austrittsstrecken und (3) die Nebenkosten aus Abschnitt 7, die bei kleinen Metrepreisen die Einheitspreissumme übersteigen können. Ein Angebot nur auf Basis von 45 €/m und 2 €/m ohne Längen und ohne Nebenkosten ist keine Kalkulation.

---

## 5. Sicherheit (kampfmittelspezifisch)

### 5.1 Grundsatz

Kampfmittelsondierung ist keine Vermessung. Jede Aussage „frei" ist eine Aussage über Menschenleben im Schacht und an der Ortsbrust. Nichts in diesem Abschnitt ersetzt die Gefährdungsbeurteilung der Fachfirma und die Vorgaben des KBD.

Regelwerke und Fundstellen [N]:
- Sprengstoffgesetz: Erlaubnis nach § 7 SprengG für die Fachfirma, Befähigungsschein nach § 20 SprengG für die Verantwortliche Person, Fachkunde nach § 9 SprengG.
- BFR KMR (Baufachliche Richtlinien Kampfmittelräumung des Bundes, aktuelle Ausgabe): Phasenmodell, Gefährdungsabschätzung, Sondierverfahren, Räumziele, Dokumentation, Freigabe. Konkrete Anhänge in der aktuellen Fassung prüfen.
- Landesrecht: Erlasse und Technische Vorgaben des zuständigen Innenministeriums bzw. KBD (in NRW: Runderlass zur Kampfmittelbeseitigung mit Technischen Verwaltungsvorschriften; in RLP: Vorgaben des Kampfmittelräumdienstes bei der ADD; im Saarland: Kampfmittelbeseitigungsdienst beim Landespolizeipräsidium). Welches Land, ist erst noch festzustellen.
- DGUV: DGUV Information 201-027 (Handlungsanleitung zur Gefährdungsbeurteilung und Festlegung von Schutzmaßnahmen bei der Kampfmittelräumung); weitere einschlägige DGUV-Regeln über BG BAU verifizieren. Baustellenverordnung (SiGeKo, SiGe-Plan mit Kampfmittelkapitel).

### 5.2 Das Bohren selbst ist ein Eingriff in nicht freigegebenen Boden [H]

Das wird in HDD-Sondierkonzepten gern übersehen: Der Pilotbohrkopf fährt durch Boden, der noch nicht sondiert ist. Trifft er einen Bombenkörper, ist eine Zündung (insbesondere bei chemischen Langzeitzündern) nicht ausgeschlossen.

- Bohransatzpunkt und Bohrgeräte-Standfläche: vor dem Aufstellen sondieren (Oberflächen- oder Bohrlochsondierung), üblicherweise ohnehin Teil der Schachtbaufeld-Freigabe.
- Bohrtrasse: Ob der KBD das Pilotbohren mit kleinem Durchmesser durch unsondierten Boden zulässt, mit welchem Personenschutz (Abstand, Abschirmung, Fernsteuerung) und ob er sondierbegleitetes Bohren fordert, ist **[N] beim zuständigen KBD zu klären und in der Gefährdungsbeurteilung zu dokumentieren.** Ich behaupte nicht, dass es zulässig ist, und nicht, dass es unzulässig ist.
- Bei Verdacht auf Kontakt während des Bohrens (plötzlicher Widerstand, Metallspäne in der Spülung, Sondenanschlag): Bohren stoppen, Gestänge nicht bewegen, Bereich räumen, Verantwortliche Person und KBD verständigen. Der Ablauf gehört schriftlich in den Notfallplan.

### 5.3 Freigabelogik und Verantwortungsübergänge [H]

Kette der Verantwortung (Annahme zur Rollenverteilung, siehe Ausgangslage):

1. **Bauherr/Auftraggeber:** Verpflichtung, den Kampfmittelverdacht abzuklären (Luftbildauswertung, historische Erkundung) und die Sondierung zu veranlassen; er formuliert das Räumziel mit dem KBD.
2. **KBD des Landes:** hoheitliche Gefahrenabwehr; gibt Verfahren, Zielobjektklasse, Sondierradius, Freigabevolumen und Berichtsformat vor; bewertet Sondierberichte; übernimmt bei Fund.
3. **Kampfmittel-Fachfirma (§ 7 SprengG):** führt die Sondierung durch, wertet aus, erstellt Sondierbericht und Freigabebescheinigung, unterschrieben von der Verantwortlichen Person. **Wer das ist, ist offen (Abschnitt 3, Punkt 1).**
4. **HDD-Firma (RTM):** stellt die Sondierachsen her, misst deren Ist-Lage ein, liefert die Bohrprotokolle. RTM gibt keine Kampfmittelfreigabe, es sei denn, RTM ist selbst Fachfirma mit § 7-Erlaubnis.
5. **Vortriebsfirma / Eiffage:** darf nur im dokumentierten **Freigabekorridor** arbeiten. Der Korridor ist ein Volumen in EPSG 31466 / EPSG 5783. Verlässt der Vortrieb den Korridor (Abweichung der Vortriebsachse), erlischt die Freigabe für den verlassenen Bereich.

Übergabepunkte, an denen etwas Schriftliches existieren muss:
- Freigabe Bohransatzpunkte und Schachtbaufelder (vor HDD).
- Ist-Lage jeder Sondierachse, eingemessen, mit Toleranzangabe (RTM an Fachfirma).
- Sondierbericht mit Anomalienliste und Deckungsnachweis: welche Volumina sind überdeckt, welche nicht (Fachfirma an AG, KBD).
- Freigabebescheinigung mit Korridorgeometrie und Auflagen (Fachfirma an AG, von dort an Eiffage/Vortrieb).
- Bestätigung, dass die geplante Vortriebsachse samt Toleranz vollständig im Korridor liegt (Vortriebsplaner).

### 5.4 Restrisiko [H]

Auch eine vollständige Sondierung liefert keine Nullgarantie. Restrisiken, die im Sondierbericht und in der Gefährdungsbeurteilung benannt werden müssen:

- Magnetometrie erfasst nur ferromagnetische Körper. Nicht oder schwach magnetische Kampfmittel bleiben unentdeckt.
- Bereiche mit Störfeldern (Spundwände der Schächte, Gleise, Oberleitungsmasten, Stahlleitungen, Bewehrung, benachbarte fertige Vortriebe) sind teilweise nicht auswertbar. Diese Bereiche sind **auszuweisen, nicht stillschweigend als frei zu behandeln.** Die ersten Meter ab Startschacht sind typischerweise betroffen.
- Deckungslücken durch Bohrabweichung (Abschnitt 1.7).
- Anomalien, die erkannt, aber nicht verifiziert oder geborgen wurden (z. B. unzugänglich unter dem Kreuzungsbauwerk), sind kein freigegebener Boden.
- Alles außerhalb des Freigabekorridors (Schmierinjektionen, Setzungsmulde, spätere Nebenbauwerke) ist nicht freigegeben.

### 5.5 Fund-Szenario während des Vortriebs [H]

Bei einem Vortrieb DN 2800 sieht niemand die Ortsbrust. Ein Fund macht sich indirekt bemerkbar (Widerstandssprung, Metallteile im Förderstrom, Blockade der Abbauwerkzeuge) oder gar nicht. Deshalb liegt das Gewicht auf der Sondierung **vor** dem Vortrieb, nicht auf der Reaktion währenddessen. Trotzdem braucht es einen Ablauf:

1. Vortrieb stoppen, Maschine nicht weiter drehen oder pressen. Kein Druckabbau, keine Spülungsänderung ohne Rücksprache.
2. Personal aus Schacht und Rohrstrang, Absperrung nach Vorgabe der Verantwortlichen Person, Bereich sichern.
3. Verantwortliche Person, KBD, Polizei verständigen. Der KBD übernimmt die Lage.
4. Erkundung von außen: vertikale Verifikationsbohrung von der Oberfläche zur Anomalie, sofern zugänglich. Unter einem Kreuzungsbauwerk ist das oft nicht möglich.
5. Optionen, die dann nur noch der KBD und der Bauherr haben: Freilegung und Bergung (Evakuierungsradius [N] nach KBD), Entschärfung oder Sprengung vor Ort, Umplanung der Trasse oder der Gradiente.
6. Vertraglich: Wer trägt Stillstand, Umplanung, Maschinenschäden? (Abschnitt 3, Punkt 13.)

Für die Bohrphase selbst gilt 5.2. Für die Sondierphase gilt: Eine erkannte Anomalie im Korridor führt zu **keiner** Freigabe des betroffenen Abschnitts, bis sie verifiziert und beseitigt oder als unschädlich identifiziert ist.

---

## 6. Nachhaltigkeit und langfristige Lösung

Was über die Bauphase hinaus tragen soll [H]:

**Dokumentation und Verwertbarkeit der Sondierdaten**
- Rohdaten der Sonde (drei Achsen, Messtakt, Weg entlang der Achse) in einem offenen Format (z. B. CSV) mit Metadaten: Sonde, Kalibrierung, Datum, Bearbeiter, Bohrung, Rohrmaterial.
- Ist-Geometrie jeder Sondierachse als 3D-Linie in EPSG 31466 / EPSG 5783 **und** in ETRS89/UTM bzw. DHHN2016 mit dokumentiertem Transformationsweg. Wer in zehn Jahren die Daten braucht, arbeitet nicht mehr in GK Zone 2.
- Freigabekorridor als Volumen (GIS-tauglich: GeoPackage/Shape/DXF mit Höhen), Anomalienliste mit Koordinaten, Bewertung und Verbleib.
- Meldung der Ergebnisse an den KBD für dessen Kataster; nicht beseitigte Verdachtspunkte ausdrücklich als solche registrieren. Das schützt spätere Bauvorhaben in der Nachbarschaft (Leitungen, Schächte, Neubauten).
- Sondierbericht als abgeschlossenes Dokument mit Deckungsnachweis, damit die Freigabe auch später prüfbar bleibt.

**Wiederauffindbarkeit**
- Ein- und Austrittspunkte der Bohrungen eingemessen; verbleibende Rohre DN 70 im Bestandsplan des Bauherrn und im Leitungskataster eintragen, auch wenn sie verfüllt sind. Ein späterer Bagger, der ein „unbekanntes Rohr" findet, löst sonst eine neue Kampfmittelfrage aus.

**Bohrkanalverdämmung**
- Jeder Bohrkanal und jedes verbleibende Rohr werden vollständig verfüllt, mit schwindarmem, gering durchlässigem Dämmer, Mengenbilanz gegen das rechnerische Bohrlochvolumen, Verfüllprotokoll je Bohrung. Unverfüllte Kanäle sind Wasserwegsamkeiten, unter Gewässern oder Dämmen ein Erosionsrisiko, im Vortriebsbereich ein Risiko für Wasserzutritt und Stützdruckverlust an der Ortsbrust.
- Bohrungen, die die Vortriebsmaschine durchfährt (Zentralbohrung, bei 3er-Anordnung alle), müssen so verfüllt sein, dass Dämmer und Rohrreste den Abbau nicht stören. Rohrmaterial danach wählen.

**Umwelt / Bohrspülung**
- Bentonitspülung ohne bedenkliche Additive; Spülungsrecycling; Entsorgung nach Abfallrecht mit Nachweis. Bohrgut aus Kampfmittelverdachtsflächen nicht ungeprüft wiederverwenden.
- Frac-out-Vermeidung (Standsicherheitsnachweis Spüldruck) mit besonderem Augenmerk auf Gewässer und Gleiskörper; Havarieplan für Spülungsaustritt.
- So wenig Fremdmaterial im Boden wie möglich: prüfen, ob die Rohre DN 70 gezogen werden können, statt 18 bis 42 Rohrstränge im Untergrund zu belassen.

**Langfristige Lösung im Sinne der Trasse**
- Der Freigabekorridor sollte so bemessen werden, dass er die spätere Betriebs- und Instandhaltungsphase abdeckt (z. B. Ringraum, spätere Injektionen), nicht nur die Maschine. Das ist eine Frage an AG und KBD, nicht an die Bohrfirma.

---

## 7. Finanzielle Aufwendung über den Metrepreis hinaus

Positionen, die 45 €/m und 2 €/m erfahrungsgemäß nicht enthalten oder die ausdrücklich zu klären sind [H]. Beträge werden bewusst nicht erfunden; Spalte „Treiber" nennt, wovon die Höhe abhängt.

| Position | Treiber | Zuordnung (Annahme) |
|---|---|---|
| Baustelleneinrichtung, Mobilisierung, Umsetzen je Kreuzung und ggf. je Vortrieb | 2 Kreuzungen, Entfernung, Zufahrten, Sperrflächen | RTM, pauschal je Kreuzung |
| Vorhaltung Bohrgerät, Spülungsanlage, Recycling, Personal in Wartezeiten | Bauzeit, Freigabeläufe, Sperrpausen | RTM; Wartezeiten vertraglich regeln |
| Messtechnik HDD: Walkover, Wireline/Gyro-Bohrlochvermessung (unter Gleis/Gewässer zwingend) | Tiefe, Kreuzungsart, Toleranzanforderung t | RTM |
| Messtechnik Sondierung: 3-Achs-Sonde, Logger, Wegmessung, Kalibrierung | Sondentyp, KBD-Anforderungen | Fachfirma (offen) |
| Sondierung, Auswertung, Sondierbericht, Freigabebescheinigung, Verantwortliche Person nach SprengG | Bohrmeter, Anzahl Anomalien, Berichtsformat des KBD | Fachfirma (offen), meist als €/m Sondierung plus Pauschale je Bericht |
| Sondierung der Bohransatzpunkte und Geräte-Standflächen | Anzahl Ansatzpunkte (bis zu 2 × 42), Verfahren | Fachfirma / AG |
| Gefährdungsbeurteilung Kampfmittel, Sicherheitskonzept, Notfallplan, Abstimmungen mit KBD | Aufwand Fachplaner | Fachfirma / eigene Firma |
| Vermessung: Absteckung, Einmessung, Transformation GK/UTM, DHHN92/2016 | Anzahl Bohrungen, Genauigkeit | RTM oder eigene Firma |
| Standsicherheitsnachweise (Spüldruck, Kreuzungsbauwerk) | Anforderungen des Kreuzungsträgers | RTM (Ingenieurleistung) |
| Projektplan, Terminsteuerung, Schnittstelle Schachtbau/Vortrieb | Komplexität, 2 Kreuzungen parallel | eigene Firma |
| Genehmigungen: Nacht-/Sonn-/Feiertagsarbeit; verkehrsrechtliche Anordnung; Betra/Sperrpausen bei Bahn; wasserrechtliche Erlaubnis bei Gewässer; Bohranzeige an die geologische Landesbehörde [N] | Kreuzungsart, Behördenlaufzeiten | eigene Firma / AG; Gebühren durchlaufend |
| Nacht-/Wochenendzuschläge Personal, Lärmschutz, Beleuchtung | Terminlage 05.10.2026, Sperrpausen | RTM und Fachfirma, Zuschlagsätze nach Tarif/eigener Kalkulation |
| Bohrspülung: Bentonit, Additive, Entsorgung mit Nachweis | Bohrmeter, Bodenart | RTM |
| Verdämmung: Dämmer je m Bohrkanal plus Rohr, Verfüllprotokolle | Bohrmeter, Bohrlochvolumen | RTM (prüfen, ob im Metrepreis) |
| Rohr DN 70: Material vs. Einbau; Ziehen statt Belassen | 2 €/m ist ein Materialpreis, kein Einbaupreis | RTM |
| Stillstand: Warten auf KBD-Freigabe, Anomalie-Verifikation, Sperrpausen, Witterung | Tage × Tagessatz Gerät und Personal | vertraglich regeln, sonst eigenes Risiko |
| Risikopositionen: Bohrabbruch (Hindernis, Verdachtskontakt), Bohrlochverlust, Nachbohrung bei Deckungslücke, Frac-out-Sanierung, Nachforderungen des KBD | Wahrscheinlichkeit × Aufwand | Angebot mit Risikozuschlag oder Eventualpositionen |
| Kampfmittelfund: Evakuierung, Bergung, Umplanung, Maschinenschaden | nicht kalkulierbar | AG / Bauherr (vertraglich festschreiben) |
| Versicherung: Deckungserweiterung Kampfmittelrisiko | Police | RTM und eigene Firma |
| Gewährleistung/Haftung für die Freigabeaussage | — | Fachfirma; nicht auf die Bohrfirma abwälzbar |

Größenordnung ohne Zahlen: Bei Metrepreisen dieser Höhe können die Nebenkosten (Mobilisierung, Vorhaltung, Sondierleistung, Genehmigungen, Zuschläge) die Einheitspreissumme aus Abschnitt 4 erreichen oder übersteigen. Ein Angebot braucht deshalb Pauschal- und Eventualpositionen neben den Metrepreisen.

---

## 8. Was sonst noch relevant ist: Widersprüche und Lücken in der Ausgangslage

1. **3 vs. 7 ist möglicherweise gar kein Fachkonflikt**, sondern „3 Vortriebe je Kreuzung" gegen „7 Bohrungen je Vortrieb" (Abschnitt 1.6). Vor der KBD-Diskussion die Originalstellen prüfen.
2. **Das LV ist ein HDD-LV, kein Kampfmittel-LV.** Genannt sind DIN 18324, DCA, GW 321, GW 301/302: alles Rohrleitungsbau. Kein Wort zu BFR KMR, KBD-Vorgaben, SprengG, Zielobjektklasse, Sondierradius, Freigabe. Die Position „Dokumentation der Sondierergebnisse" steht allein. Wer die eigentliche Sondierung schuldet, ist im LV nicht definiert.
3. **DWA-A 125 und „Anfahr-/Einfahrphasen" stammen aus der Vortriebswelt.** Entweder wurde Text aus dem Vortriebs-LV übernommen oder RTM soll Leistungen des Vortriebs mit übernehmen. Beides muss vor Angebotsabgabe eindeutig sein, sonst schuldet man am Ende Standsicherheitsnachweise für einen DN 2800-Vortrieb zu 45 €/m Bohrung.
4. **Baubeginn 05.10.2026 bei Stand 17.09.2026.** Ohne Klärung des Begriffs ist der Termin entweder harmlos oder unmöglich (Abschnitt 3, Punkt 3).
5. **Preise ohne Mengen.** 45 €/m und 2 €/m ohne Längen, ohne Leistungsumfang, ohne Angabe, ob RTM-Angebot oder Eiffage-Zielpreis. Ein Preisvergleich mit der eigenen Kalkulation ist erst nach Klärung von Abschnitt 3, Punkt 6 möglich.
6. **Rohr „DN 70"** ist keine Standard-Nennweite der üblichen PE-/PVC-Reihen; wahrscheinlich d 75. Innendurchmesser gegen Sondendurchmesser prüfen.
7. **Koordinatenbezug ist veraltet** (DHDN/GK Zone 2, DHHN92 statt ETRS89/UTM, DHHN2016). Möglich, dass der AG Altpläne in GK führt. Transformationsweg und Verantwortung festlegen; sonst dm-Fehler in einem System, das mit dm-Reserven arbeitet.
8. **Kreuzungsart unbekannt.** Straße, Bahn, Gewässer oder Damm bestimmen Genehmigungen, Ortungsverfahren, Standsicherheitsnachweise und Fund-Szenario. Aus der Mail nicht ersichtlich.
9. **Skizze fehlt in der Weiterleitung** (jedenfalls für diese Analyse). Ohne Skizze ist die Deckungsrechnung nur parametrisch. Mit Skizze ist sie in einer Stunde konkret.
10. **Bohrungen zwischen benachbarten Vortrieben** könnten doppelt genutzt werden (3 Vortriebe je Kreuzung). Das kann die 7er-Variante spürbar verbilligen, ist aber nur mit der Skizze und den Achsabständen rechenbar.
11. **Reihenfolge und Störfelder:** Ein fertiger Vortrieb (Bewehrung, ggf. Stahlrohr) neben einer noch zu sondierenden Achse stört die Messung. Alle Sondierbohrungen einer Kreuzung vor dem ersten Vortrieb dort.
12. **Alternative Verfahren** (Sondierung aus dem Startschacht heraus mit horizontalen Kernbohrungen, Sondierung aus der Vortriebsmaschine bei Stillstand) wurden hier nicht bewertet, weil die Ausgangslage HDD vorgibt. Sollte der KBD das HDD-Konzept nicht akzeptieren (5.2), müssen sie auf den Tisch.
13. **Zuständigkeit für die Freigabe** ist der einzige Punkt, ohne den alle anderen belanglos sind. Er sollte am Anfang des Gesprächs mit Rai stehen.

---

## Anhang: Formeln zum Nachrechnen mit den Ist-Werten der Skizze

- Freigaberadius: R_F = D_a / 2 + s
- Wirksamer Sondierradius: r_eff = r_d − t
- 1 Bohrung: R_F ≤ r_eff
- 3 Bohrungen (120°, Achsabstand 0,5 · R_F vom Zentrum): R_F ≤ r_eff / 0,866
- 7 Bohrungen (1 + 6, äußere im Abstand 0,866 · R_F): R_F ≤ 2 · r_eff
- Beliebige Anordnung: für jeden Punkt des Freigabekreises muss der Abstand zur nächsten eingemessenen Bohrachse ≤ r_eff sein; KBD-Überlappung zusätzlich einhalten.
- Bohrlänge bei Oberflächenstart: L_Bohrung ≈ L_Vortrieb + 2 · (h / tan α + Bogenanteil) + Überlappung an den Schächten
- Summe Einheitspreise: n_Vortriebe × n_Bohrungen × L_Bohrung × (45 + 2) €/m

Alle Eingangswerte außer den Einheitspreisen und den Stückzahlen 6 / 3 / 7 sind in diesem Dokument Annahmen.
