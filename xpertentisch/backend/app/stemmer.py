"""Ein kleiner deutscher Stemmer — regelbasiert, deterministisch, ohne Abhängigkeit.

Warum es ihn gibt: die Themen-Linse ist nur so gut wie die Begriffe darunter.
Ohne Stammformen sind „Sicherung" und „Sicherungen" zwei Begriffe, „Ausfall"
und „Ausfälle" ebenso — und jeder von ihnen wiegt halb so schwer, wie er
sollte. Zusammengeführt wird hier deshalb die **Beugung**, nicht die Bedeutung.

Was er tut und was ausdrücklich nicht:

* Er schneidet Beugungsendungen ab (Plural, Fälle, Personalformen) — die
  Schritte 1 und 2 des bekannten Snowball-Verfahrens für das Deutsche.
* Er lässt die **Ableitungssilben** stehen: „-ung", „-heit", „-keit", „-lich",
  „-isch" bleiben. Snowball würde „Sicherung" auf „sicher" kürzen und damit
  das Hauptwort mit dem Eigenschaftswort verschmelzen. Für eine Anwendung, die
  nichts behaupten will, ist das eine Zusammenlegung zu viel: „Sicherung" und
  „sicher" sind nicht dasselbe Thema.
* Er schlägt nichts nach. Es gibt keine Wörterliste, kein Modell, keinen
  Netzzugriff — dieselbe Eingabe ergibt immer dieselbe Ausgabe.

Der Stamm ist ein **Vergleichsschlüssel**, keine Beschriftung. Angezeigt wird
immer die Form, die tatsächlich im Text steht; welche das ist, entscheidet
`app.analysis`. So sieht niemand je ein abgeschnittenes Wort auf dem Schirm.
"""

from __future__ import annotations

VOKALE = frozenset("aeiouyäöü")

#: Nach diesen Buchstaben darf ein alleinstehendes „s" abgeschnitten werden.
#: (Snowball: valid s-ending. Sonst verlöre „das" oder „was" seinen Sinn.)
S_ENDUNG = frozenset("bdfghklmnrt")
#: Nach diesen Buchstaben darf „st" abgeschnitten werden.
ST_ENDUNG = frozenset("bdfghklmnt")

#: Kürzer als das rühren wir ein Wort nicht an — sonst bleibt kein Stamm übrig.
MINDESTLAENGE = 4
#: So viele Zeichen müssen vor der abgeschnittenen Endung stehen bleiben.
MINDESTREST = 3


def _vorspiel(wort: str) -> str:
    """Vereinheitlicht, was nur Schreibweise ist.

    ß wird zu ss, Umlaute werden aufgelöst. Damit fallen „Ausfälle" und
    „Ausfall" nach dem Endungsschnitt zusammen — der Plural ist dort im
    Umlaut versteckt, nicht nur in der Endung.
    """
    return (
        wort.replace("ß", "ss")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
    )


def _r1(wort: str) -> int:
    """Beginn des Bereichs R1: hinter dem ersten Nicht-Vokal nach einem Vokal.

    Endungen werden nur innerhalb von R1 geschnitten. Das ist die Regel, die
    verhindert, dass aus kurzen Wörtern Buchstabenreste werden. Snowball zieht
    R1 zusätzlich auf mindestens 3 vor.
    """
    grenze = len(wort)
    for i in range(1, len(wort)):
        if wort[i] not in VOKALE and wort[i - 1] in VOKALE:
            grenze = i + 1
            break
    return max(grenze, MINDESTREST)


def _schneide(wort: str, endung: str, r1: int) -> str | None:
    """Schneidet eine Endung ab, wenn sie ganz in R1 liegt und genug übrig bleibt."""
    if not wort.endswith(endung):
        return None
    rest = len(wort) - len(endung)
    if rest < r1 or rest < MINDESTREST:
        return None
    return wort[:rest]


def stamm(wort: str) -> str:
    """Der Vergleichsschlüssel eines Wortes.

    Idempotent: ``stamm(stamm(w)) == stamm(w)`` — geprüft in den Tests, weil
    sonst die Themen-Linse je nach Aufrufreihenfolge andere Gruppen bildete.
    """
    w = _vorspiel(wort.lower())
    if len(w) < MINDESTLAENGE:
        return w
    r1 = _r1(w)

    # Schritt 1: Plural- und Fallendungen.
    for endung in ("ern", "em", "er", "en", "es", "e"):
        gekuerzt = _schneide(w, endung, r1)
        if gekuerzt is not None:
            w = gekuerzt
            # „-nis" wird im Plural zu „-nisse": das doppelte s muss weg.
            if w.endswith("niss"):
                w = w[:-1]
            break
    else:
        # Ein einzelnes „s" nur nach einem Buchstaben, der es tragen kann.
        if w.endswith("s") and len(w) >= 2 and w[-2] in S_ENDUNG:
            gekuerzt = _schneide(w, "s", r1)
            if gekuerzt is not None:
                w = gekuerzt

    # Schritt 2: Personal- und Steigerungsendungen.
    for endung in ("est", "er", "en"):
        gekuerzt = _schneide(w, endung, r1)
        if gekuerzt is not None:
            w = gekuerzt
            break
    else:
        if w.endswith("st") and len(w) >= 3 and w[-3] in ST_ENDUNG:
            gekuerzt = _schneide(w, "st", r1)
            # Snowball verlangt hier zusätzlich Platz für den Wortkern.
            if gekuerzt is not None and len(gekuerzt) >= MINDESTREST:
                w = gekuerzt

    return w
