#!/usr/bin/env python3
"""
build_personen_bibliothek.py — erzeugt die Daten der TiSCH Personen-Bibliothek.

Quelle (nur zur BUILD-Zeit gelesen):
    /Volumes/ThunderBolt4_2TB/MeineApps/shared-core/data/personalities.json
Ausgabe (Laufzeit-Artefakt, KEINE /Volumes-Abhängigkeit):
    der-tisch-backend/data/personen_bibliothek.json

Pro Person wird die kurze, geprüfte `bio` aus dem shared-core unverändert als
Kern übernommen. `beschreibung` und `kernideen` sind ERGÄNZT und ausdrücklich
MODELLGENERIERT (`beschreibung_quelle: "model_generated"`) — kein geprüftes
Wissen, sondern ausführlicher erläuternder Text. Ebenso die Disziplin-Texte.

Regenerieren:
    python3 der-tisch-backend/data/build_personen_bibliothek.py
Fehlt die /Volumes-Quelle, bleibt eine vorhandene Ausgabedatei unangetastet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SHARED_CORE = Path(
    "/Volumes/ThunderBolt4_2TB/MeineApps/shared-core/data/personalities.json"
)
OUT = Path(__file__).parent / "personen_bibliothek.json"


# ---------------------------------------------------------------------------
# Disziplin-Beschreibungen (modellgeneriert)
# ---------------------------------------------------------------------------
DISZIPLINEN = {
    "Upanishadische Tradition": (
        "Die upanishadische Tradition Indiens fragt nach dem Verhältnis von "
        "individuellem Selbst (Atman) und Weltgrund (Brahman). Aus den "
        "Upanishaden entwickelt sich über Jahrhunderte das Vedanta — eine "
        "Familie von Schulen, die Erkenntnis nicht als Information, sondern "
        "als befreiende Selbsterkenntnis versteht."
    ),
    "Buddhismus": (
        "Der Buddhismus geht auf die Einsichten Siddhartha Gautamas zurück "
        "und versteht Leiden nicht als Schicksal, sondern als analysierbaren "
        "Prozess. Statt nach einem Schöpfergott fragt er nach den Bedingungen "
        "von Bindung und Befreiung — und entfaltet über die Jahrhunderte "
        "Schulen von strenger Analyse bis zu lebendiger Achtsamkeitspraxis."
    ),
    "Daoismus": (
        "Der Daoismus sucht nicht die Beherrschung der Welt, sondern den "
        "Einklang mit dem Dao — dem unergründlichen „Weg“, dem alles folgt. "
        "Sein Ideal ist das Wu Wei, das mühelose Handeln, das sich gegen "
        "Zwang und Übermaß stellt und im Natürlichen die größere Weisheit "
        "erkennt."
    ),
    "Griechische Schule": (
        "Mit der klassischen griechischen Philosophie verschiebt sich das "
        "Fragen vom Mythos zum Begriff. In Athen entstehen mit Sokrates und "
        "Platon Schulen, die Erkenntnis, Tugend und gerechtes Zusammenleben "
        "zum prüfbaren Gegenstand des Denkens machen."
    ),
    "Aristotelische Schule": (
        "Mit Aristoteles wird die Philosophie zur ordnenden Wissenschaft. "
        "Statt einer Welt der Urbilder untersucht er das Einzelne selbst, "
        "seine Ursachen und Zwecke — und begründet damit Logik, "
        "Naturforschung und Ethik als getrennte, methodische Disziplinen."
    ),
    "Vorsokratik": (
        "Die Vorsokratiker fragen zum ersten Mal systematisch nach Ursprung "
        "und Bauprinzip der Welt — nicht mehr in Göttergeschichten, sondern "
        "in Begriffen wie Werden, Sein und Zahl. Ihre knappen, oft "
        "rätselhaften Fragmente stehen am Anfang des abendländischen Denkens."
    ),
    "Stoische Linie": (
        "Die Stoa lehrt, zwischen dem zu unterscheiden, was in unserer Macht "
        "steht, und dem, was nicht — und allein den eigenen Urteilen und "
        "Haltungen Sorge zu widmen. Gelassenheit ist hier kein Rückzug, "
        "sondern eine eingeübte Kunst, inmitten von Schicksal und Pflicht "
        "frei zu bleiben."
    ),
    "Christliche Theologie": (
        "Die christliche Theologie und Mystik fragt, wie der Mensch dem "
        "unverfügbaren Gott begegnet — in der Innerlichkeit, in der Vision, "
        "im Loslassen. Sie verbindet Denken und Glauben zu einem Weg der "
        "Verwandlung, der bis an die Grenzen der Sprache führt."
    ),
    "Konfuzianismus": (
        "Der Konfuzianismus fragt, wie der Mensch durch Bildung, Ritual und "
        "Selbstkultivierung zu einem guten Glied einer guten Ordnung wird. "
        "Ethik ist hier nicht Theorie, sondern eingeübte Haltung — getragen "
        "von Menschlichkeit (Ren) und dem rechten Maß im Umgang."
    ),
    "Neuplatonismus": (
        "Der Neuplatonismus deutet die Wirklichkeit als gestufte Ordnung, "
        "die aus einem letzten Ursprung — dem „Einen“ — hervorgeht und in es "
        "zurückstrebt. Philosophie wird hier zum Aufstieg der Seele durch "
        "die Ebenen des Seins."
    ),
    "Neuzeit": (
        "Die neuzeitliche Philosophie verschiebt den Grund der Gewissheit in "
        "das denkende, kritische Subjekt. Von Descartes’ Zweifel bis zu "
        "Nietzsches Umwertung wird das Denken selbst zum Prüfstein — und "
        "zugleich zur offenen Frage."
    ),
}


# ---------------------------------------------------------------------------
# Erweiterte Personen-Beschreibungen (modellgeneriert)
# ---------------------------------------------------------------------------
EXPANDED = {
    "personality-yajnavalkya": {
        "beschreibung": (
            "Yajnavalkya gilt als einer der frühesten namentlich fassbaren "
            "Denker Indiens und prägt große Teile der Brihadaranyaka-"
            "Upanishad. In berühmten Streitgesprächen — etwa mit der "
            "Philosophin Gargi — bestimmt er das Selbst (Atman) negativ: "
            "„neti, neti“, nicht dies, nicht dies. Das wahre Selbst ist "
            "reines Bewusstsein, das alles erkennt und selbst nie zum "
            "Objekt wird. Damit legt er den Grundstein der gesamten "
            "späteren Vedanta-Philosophie."
        ),
        "kernideen": [
            "Atman als reines, objektloses Bewusstsein",
            "„neti, neti“ — Wahrheit über die Verneinung",
            "Frühe dialogische Philosophie",
        ],
    },
    "personality-patanjali": {
        "beschreibung": (
            "Patanjali bündelt in den Yoga-Sutras eine bis dahin verstreute "
            "Praxis zu einem klaren System. Sein achtgliedriger Pfad "
            "(Ashtanga) führt von ethischen Grundregeln über Körperhaltung "
            "und Atem bis zu Sammlung und Versenkung. Yoga definiert er als "
            "das „Zur-Ruhe-Kommen der Bewegungen des Geistes“. Sein Werk ist "
            "bis heute die maßgebliche Referenz für Meditationspraxis im "
            "Hinduismus."
        ),
        "kernideen": [
            "Achtgliedriger Pfad des Yoga",
            "Yoga als Stillwerden des Geistes",
            "Systematik statt loser Praxis",
        ],
    },
    "personality-shankara": {
        "beschreibung": (
            "Shankara gilt als der bedeutendste Philosoph des Advaita "
            "Vedanta, der „Nicht-Zweiheit“. Er lehrt, dass Atman und "
            "Brahman letztlich identisch sind und die Vielheit der Welt auf "
            "Unwissenheit (Avidya) beruht. Mit scharfen Kommentaren zu "
            "Upanishaden, Bhagavadgita und Brahmasutras und der Gründung von "
            "Klöstern in allen Himmelsrichtungen formte er den Hinduismus "
            "dauerhaft — und das in einem kurzen Leben."
        ),
        "kernideen": [
            "Identität von Atman und Brahman",
            "Welt-Vielheit als Avidya (Unwissenheit)",
            "Nicht-Dualität (Advaita)",
        ],
    },
    "personality-ramana": {
        "beschreibung": (
            "Ramana Maharshi durchlebte als Jugendlicher eine spontane "
            "Todeserfahrung, aus der eine bleibende Gewissheit des Selbst "
            "hervorging. Seine zentrale Methode ist die Selbsterforschung: "
            "die beharrliche Frage „Wer bin ich?“, die den Geist zu seiner "
            "Quelle zurückführt. Er lehrte ohne Rituale, Dogmen oder "
            "Systemzwang, oft im Schweigen. Vom Berg Arunachala aus wurde er "
            "zu einer der einflussreichsten spirituellen Gestalten des "
            "modernen Indien."
        ),
        "kernideen": [
            "Selbsterforschung: „Wer bin ich?“",
            "Lehre jenseits von Ritual und Dogma",
            "Das Schweigen als Unterweisung",
        ],
    },
    "personality-vivekananda": {
        "beschreibung": (
            "Swami Vivekananda, Schüler des Mystikers Ramakrishna, trug das "
            "Vedanta 1893 auf das Weltparlament der Religionen in Chicago "
            "und wurde zur Stimme eines selbstbewussten, dialogfähigen "
            "Hinduismus. Er verband spirituelle Tiefe mit sozialem "
            "Engagement und gründete den Ramakrishna-Orden. Religion war für "
            "ihn nicht Lehrsatz, sondern Verwirklichung des Göttlichen im "
            "Menschen."
        ),
        "kernideen": [
            "Vedanta im globalen Dialog",
            "Spiritualität verbunden mit sozialem Dienst",
            "Religion als Verwirklichung, nicht Dogma",
        ],
    },
    "personality-buddha": {
        "beschreibung": (
            "Siddhartha Gautama gab als Sohn eines Adelsgeschlechts ein "
            "Leben im Wohlstand auf, um dem Leiden auf den Grund zu gehen. "
            "Nach Jahren der Askese fand er den „Mittleren Weg“ zwischen "
            "Selbstkasteiung und Genuss und formulierte die Vier Edlen "
            "Wahrheiten. Sein Weg zielt nicht auf Glauben, sondern auf "
            "eigene Einsicht und das Erlöschen von Gier, Hass und "
            "Verblendung."
        ),
        "kernideen": [
            "Vier Edle Wahrheiten",
            "Der Mittlere Weg",
            "Befreiung durch Einsicht statt Glauben",
        ],
    },
    "personality-nagarjuna": {
        "beschreibung": (
            "Nagarjuna begründete die Madhyamaka-Schule, die „Mittlere "
            "Lehre“, und gilt nach dem Buddha als wichtigster Denker des "
            "Buddhismus. Sein Schlüsselbegriff ist Shunyata — die Leerheit: "
            "Nichts existiert aus sich selbst, alles entsteht in "
            "Abhängigkeit. Mit einer radikalen Dialektik zeigt er die "
            "Widersprüche jeder festgelegten Position auf, ohne selbst eine "
            "neue These zu behaupten."
        ),
        "kernideen": [
            "Shunyata — Leerheit aller Dinge",
            "Abhängiges Entstehen",
            "Dialektik ohne eigene Letzt-These",
        ],
    },
    "personality-thich-nhat-hanh": {
        "beschreibung": (
            "Der vietnamesische Zen-Meister Thich Nhat Hanh prägte den "
            "Begriff des „engagierten Buddhismus“ — Meditation, die sich dem "
            "Leid der Welt nicht entzieht. Sein Friedensengagement im "
            "Vietnamkrieg zwang ihn ins jahrzehntelange Exil. In einer "
            "einfachen, poetischen Sprache machte er Achtsamkeit für ein "
            "westliches Publikum zugänglich: bewusstes Atmen, bewusstes "
            "Gehen, Präsenz im Alltäglichen."
        ),
        "kernideen": [
            "Engagierter Buddhismus",
            "Achtsamkeit im Alltag",
            "Frieden beginnt im gegenwärtigen Atemzug",
        ],
    },
    "personality-goenka": {
        "beschreibung": (
            "S. N. Goenka, ein in Burma geborener Geschäftsmann indischer "
            "Herkunft, lernte Vipassana bei dem Lehrer Sayagyi U Ba Khin und "
            "gab sie ab 1969 weltweit weiter. In zehntägigen, kostenfreien "
            "Retreats vermittelt er eine nicht-konfessionelle Technik der "
            "Körperbeobachtung: das gleichmütige Wahrnehmen von "
            "Empfindungen, das tief sitzende Reaktionsmuster auflöst."
        ),
        "kernideen": [
            "Vipassana als Körperbeobachtung",
            "Gleichmut gegenüber allen Empfindungen",
            "Nicht-konfessionelle, freie Retreats",
        ],
    },
    "personality-ajahn-chah": {
        "beschreibung": (
            "Ajahn Chah war ein thailändischer Waldmönch, der die strenge, "
            "einfache Tradition des Theravada-Waldbuddhismus erneuerte. Sein "
            "Kloster zog auch westliche Schüler an, aus denen eine "
            "internationale Klosterlinie erwuchs. Seine Belehrungen waren "
            "konkret, humorvoll und unprätentiös — Gelassenheit lernte man "
            "bei ihm an Alltagspflichten und am genauen Hinsehen, nicht an "
            "Theorie."
        ),
        "kernideen": [
            "Theravada-Waldtradition",
            "Lehre durch Einfachheit und Humor",
            "Praxis im konkreten Alltag",
        ],
    },
    "personality-laozi": {
        "beschreibung": (
            "Laozi gilt als legendärer Verfasser des Daodejing, eines der "
            "meistübersetzten Bücher der Welt. Ob historische Person oder "
            "Sammelgestalt — sein Werk lehrt das Dao, den „Weg“, der sich "
            "jeder Definition entzieht, und das Wu Wei, das nicht-"
            "erzwingende Handeln. Stärke zeigt sich für ihn im Weichen, "
            "Führung im Zurücktreten."
        ),
        "kernideen": [
            "Das Dao jenseits aller Begriffe",
            "Wu Wei — Handeln ohne Erzwingen",
            "Die Stärke des Weichen",
        ],
    },
    "personality-zhuangzi": {
        "beschreibung": (
            "Zhuangzi ist der große Stilist und Provokateur des Daoismus. "
            "In Gleichnissen, Paradoxien und absurden Geschichten — am "
            "bekanntesten der Traum vom Schmetterling — relativiert er jeden "
            "festen Standpunkt und jede Gewissheit. Er verspottet Ehrgeiz "
            "und Karriere und preist die Freiheit dessen, der „nutzlos“ und "
            "so ungebunden ist."
        ),
        "kernideen": [
            "Relativität aller Standpunkte",
            "Der Schmetterlingstraum",
            "Freiheit durch „Nutzlosigkeit“",
        ],
    },
    "personality-zhang-sanfeng": {
        "beschreibung": (
            "Zhang Sanfeng ist eine halb legendäre Gestalt, ein "
            "daoistischer Mönch, dem die Überlieferung die Erfindung des "
            "Taiji Quan zuschreibt. Die Legende erzählt, er habe den Kampf "
            "einer Schlange mit einem Kranich beobachtet und daraus ein "
            "Prinzip gewonnen: das Weiche überwindet das Harte. Er "
            "verkörpert die Verbindung von daoistischer Philosophie, innerer "
            "Alchemie und Bewegungskunst."
        ),
        "kernideen": [
            "Taiji als Bewegung gewordene Daoistik",
            "Das Weiche überwindet das Harte",
            "Innere Alchemie und Körperpraxis",
        ],
    },
    "personality-sokrates": {
        "beschreibung": (
            "Sokrates schrieb kein Wort und wurde doch zum Wendepunkt der "
            "Philosophie. Auf den Plätzen Athens prüfte er im Gespräch die "
            "Selbstverständlichkeiten seiner Mitbürger und entlarvte "
            "Scheinwissen — wissend, dass er nichts wisse. Sein Ziel war "
            "nicht Rechthaben, sondern die gemeinsame Sorge um ein gutes, "
            "geprüftes Leben. Für seine Unbeirrbarkeit nahm er das "
            "Todesurteil in Kauf."
        ),
        "kernideen": [
            "Dialogische Prüfung (Elenchos)",
            "Das Wissen des Nichtwissens",
            "Das ungeprüfte Leben ist nicht lebenswert",
        ],
    },
    "personality-platon": {
        "beschreibung": (
            "Platon, Schüler des Sokrates, gründete in Athen die Akademie — "
            "die erste dauerhafte philosophische Schule. In kunstvollen "
            "Dialogen entwickelte er die Ideenlehre: Hinter der wandelbaren "
            "Sinnenwelt liegt eine Ordnung unwandelbarer Urbilder. Sein "
            "Höhlengleichnis prägt bis heute das Bild von Bildung als Weg "
            "ans Licht; in der „Politeia“ verband er Erkenntnis, Seele und "
            "gerechten Staat."
        ),
        "kernideen": [
            "Ideenlehre — Urbilder hinter den Erscheinungen",
            "Das Höhlengleichnis",
            "Erkenntnis, Seele und Gerechtigkeit",
        ],
    },
    "personality-aristoteles": {
        "beschreibung": (
            "Aristoteles, zwanzig Jahre Schüler Platons, wandte sich von "
            "der Ideenlehre ab und der konkreten Wirklichkeit zu. Er "
            "begründete die formale Logik, ordnete das Wissen in "
            "eigenständige Disziplinen und legte Grundlagen von Biologie, "
            "Physik und Staatslehre. Seine Tugendethik sucht das gelingende "
            "Leben (Eudaimonia) im Maß, in der „Mitte“ zwischen den "
            "Extremen."
        ),
        "kernideen": [
            "Begründung der formalen Logik",
            "Tugend als Mitte zwischen Extremen",
            "Eudaimonia — gelingendes Leben",
        ],
    },
    "personality-heraklit": {
        "beschreibung": (
            "Heraklit aus Ephesos, der „Dunkle“ genannt, denkt die Welt als "
            "unaufhörlichen Fluss: alles fließt, niemand steigt zweimal in "
            "denselben Fluss. Wirklichkeit ist für ihn Spannung der "
            "Gegensätze, gehalten von einem verborgenen Logos, der "
            "Weltvernunft. In dunklen, aphoristischen Sätzen formuliert er "
            "eine Philosophie des Werdens."
        ),
        "kernideen": [
            "„Panta rhei“ — alles fließt",
            "Einheit der Gegensätze",
            "Der Logos als Weltordnung",
        ],
    },
    "personality-parmenides": {
        "beschreibung": (
            "Parmenides aus Elea stellt dem Fluss Heraklits die radikale "
            "Gegenthese entgegen: Das Sein ist, das Nichtsein ist nicht — "
            "und folglich kann es kein echtes Werden, keine Vielheit, keine "
            "Bewegung geben. Wahrheit erschließt sich nur dem Denken, nicht "
            "den trügerischen Sinnen. Damit begründet er die Ontologie."
        ),
        "kernideen": [
            "Das Sein ist, Nichtsein ist nicht",
            "Wandel als Sinnentrug",
            "Vorrang des Denkens vor der Wahrnehmung",
        ],
    },
    "personality-pythagoras": {
        "beschreibung": (
            "Pythagoras verband mathematische Einsicht mit religiöser "
            "Lebensform. In seiner Gemeinschaft in Süditalien galt die Zahl "
            "als Schlüssel zur Ordnung des Kosmos — von den Verhältnissen "
            "der Töne bis zu den Bahnen der Gestirne. Lehren von "
            "Seelenwanderung und der „Harmonie der Sphären“ machten seine "
            "Schule zugleich zur Forschungsstätte und zum Lebensbund."
        ),
        "kernideen": [
            "Die Zahl als Ordnung des Kosmos",
            "Harmonie der Sphären",
            "Philosophie als Lebensform",
        ],
    },
    "personality-seneca": {
        "beschreibung": (
            "Seneca war Philosoph, Dramatiker und einer der mächtigsten "
            "Männer Roms — Erzieher und Berater des Kaisers Nero. In Briefen "
            "und Essays machte er die stoische Lebenskunst praktisch: über "
            "Zeit, Zorn, Tod und das rechte Maß. Sein Leben blieb "
            "spannungsvoll zwischen Reichtum, Macht und philosophischem "
            "Anspruch."
        ),
        "kernideen": [
            "Stoische Lebenskunst im Alltag",
            "Umgang mit Zeit, Zorn und Tod",
            "Philosophie zwischen Macht und Maß",
        ],
    },
    "personality-aurelius": {
        "beschreibung": (
            "Marc Aurel regierte zwei Jahrzehnte als römischer Kaiser — und "
            "führte zugleich ein philosophisches Tagebuch, die "
            "„Selbstbetrachtungen“, nicht zur Veröffentlichung, sondern zur "
            "eigenen Prüfung. Darin ringt er, mitten in Krieg und Pflicht, "
            "um stoische Haltung: Gelassenheit gegenüber dem Unabänderlichen "
            "und das Bewusstsein der Vergänglichkeit."
        ),
        "kernideen": [
            "Selbstprüfung als tägliche Übung",
            "Gelassenheit gegenüber dem Unabänderlichen",
            "Pflicht und Vergänglichkeit",
        ],
    },
    "personality-epictet": {
        "beschreibung": (
            "Epiktet begann sein Leben als Sklave und wurde zum "
            "einflussreichsten Lehrer der späten Stoa. Sein Kerngedanke: "
            "Nicht die Dinge beunruhigen uns, sondern unsere Urteile über "
            "sie. Frei wird, wer klar trennt zwischen dem, was in seiner "
            "Macht steht — die eigenen Vorstellungen und Entscheidungen — "
            "und dem, was es nicht ist."
        ),
        "kernideen": [
            "Die Dichotomie der Kontrolle",
            "Nicht die Dinge, sondern die Urteile beunruhigen",
            "Innere Freiheit unter allen Umständen",
        ],
    },
    "personality-zenon": {
        "beschreibung": (
            "Zenon von Kition, ein Kaufmann zypriotischer Herkunft, kam "
            "durch einen Schiffbruch zur Philosophie und gründete um "
            "300 v. Chr. in Athen die Stoa. Ihren Namen trägt sie von der "
            "„bunten Säulenhalle“ (Stoa Poikile). Er entwarf eine "
            "Philosophie, die Logik, Naturlehre und Ethik verband und das "
            "Leben „in Übereinstimmung mit der Natur“ zum Ziel erklärte."
        ),
        "kernideen": [
            "Gründung der Stoa",
            "Leben im Einklang mit der Natur",
            "Einheit von Logik, Physik und Ethik",
        ],
    },
    "personality-augustinus": {
        "beschreibung": (
            "Augustinus, von der antiken Rhetorik herkommend und lange "
            "suchend, fand erst spät zum christlichen Glauben — ein Weg, den "
            "er in den „Bekenntnissen“ als erste große Autobiografie der "
            "Innerlichkeit erzählt. Er entdeckt das Innere als Ort, an dem "
            "der Mensch Gott begegnet, und denkt das Verhältnis von Wille, "
            "Gnade und Freiheit neu."
        ),
        "kernideen": [
            "Innerlichkeit als Ort der Wahrheit",
            "Erinnerung, Wille und Gnade",
            "„Bekenntnisse“ — Autobiografie der Seele",
        ],
    },
    "personality-eckhart": {
        "beschreibung": (
            "Meister Eckhart, Dominikaner und Prediger, gilt als der "
            "kühnste Mystiker des deutschen Mittelalters. Er lehrt die "
            "„Gelassenheit“ — das Loslassen aller Bilder, Wünsche und "
            "Sicherheiten, bis hin zum Loslassen der eigenen "
            "Gottesvorstellungen. Seine paradoxe, an die Grenzen der "
            "Sprache gehende Lehre brachte ihm einen Häresieprozess ein."
        ),
        "kernideen": [
            "Gelassenheit — Loslassen aller Bilder",
            "Der Seelengrund",
            "Mystik an der Grenze der Sprache",
        ],
    },
    "personality-teresa": {
        "beschreibung": (
            "Teresa von Avila war Karmelitin, Mystikerin und tatkräftige "
            "Ordensreformerin im Spanien des 16. Jahrhunderts. In der "
            "„Inneren Burg“ beschreibt sie den Weg der Seele als "
            "Durchschreiten von sieben „Wohnungen“, von der ersten Umkehr "
            "bis zur Einung mit Gott. Ihre Mystik bleibt nüchtern und "
            "prüfend, misstrauisch gegen bloße Gefühle."
        ),
        "kernideen": [
            "Die „Innere Burg“ — sieben Wohnungen der Seele",
            "Mystik mit nüchterner Prüfung",
            "Kontemplation und tätige Reform",
        ],
    },
    "personality-hildegard": {
        "beschreibung": (
            "Hildegard von Bingen war Benediktinerin, Äbtissin und eine der "
            "vielseitigsten Gestalten des Mittelalters: Visionärin, "
            "Komponistin, Heilkundige. Ihre Visionen hielt sie in Wort und "
            "leuchtenden Bildern fest. Sie verband eine kosmische Schau der "
            "Schöpfung mit konkretem Wissen über Pflanzen und Körper — das "
            "„Grünen“ (viriditas) war ihr Bild lebendiger Kraft."
        ),
        "kernideen": [
            "Visionäre Schau der Schöpfung",
            "Viriditas — die „Grünkraft“ des Lebendigen",
            "Einheit von Mystik, Musik und Heilkunde",
        ],
    },
    "personality-merton": {
        "beschreibung": (
            "Thomas Merton, US-amerikanischer Trappistenmönch und "
            "Schriftsteller, wurde mit seiner Autobiografie zum "
            "meistgelesenen geistlichen Autor seiner Zeit. Aus der Stille "
            "des Klosters heraus suchte er den Dialog: mit dem Zen-"
            "Buddhismus, dem Daoismus, der Friedensbewegung. Kontemplation "
            "war für ihn kein Rückzug, sondern führte zur Verantwortung für "
            "die Welt."
        ),
        "kernideen": [
            "Kontemplation und Weltverantwortung",
            "Interreligiöser Dialog mit Zen und Daoismus",
            "Stille als Quelle des Engagements",
        ],
    },
    "personality-konfuzius": {
        "beschreibung": (
            "Konfuzius lebte in einer Zeit politischen Zerfalls und suchte "
            "die Erneuerung der Ordnung nicht in Gesetz und Strafe, sondern "
            "in der sittlichen Bildung des Einzelnen. Sein Schlüsselbegriff "
            "ist Ren — Menschlichkeit —, eingeübt in Ritual (Li), Pietät "
            "und beständigem Lernen. Der „Edle“ ist nicht von Geburt, "
            "sondern durch Selbstformung edel."
        ),
        "kernideen": [
            "Ren — Menschlichkeit als Kern der Ethik",
            "Ritual und Selbstkultivierung",
            "Der „Edle“ durch Bildung, nicht Geburt",
        ],
    },
    "personality-mengzi": {
        "beschreibung": (
            "Mengzi (Menzius) gilt als der „zweite Weise“ des "
            "Konfuzianismus. Gegen pessimistische Zeitgenossen verteidigt "
            "er die These von der angeborenen Güte des Menschen: In jedem "
            "liegen „Keime“ der Tugend — etwa das spontane Mitleid beim "
            "Anblick eines Kindes in Gefahr. Auch Herrschaft bindet er an "
            "das Wohl des Volkes."
        ),
        "kernideen": [
            "Angeborene Güte des Menschen",
            "Die „Keime“ der Tugend",
            "Herrschaft im Dienst des Volkes",
        ],
    },
    "personality-zhu-xi": {
        "beschreibung": (
            "Zhu Xi systematisierte im 12. Jahrhundert die konfuzianische "
            "Tradition neu und schuf den Neo-Konfuzianismus, der für "
            "Jahrhunderte das geistige Fundament Chinas wurde. Er verband "
            "Ethik mit einer Metaphysik von Prinzip (Li) und Lebensenergie "
            "(Qi). Erkenntnis bedeutet ihm das „Ergründen der Dinge“ — "
            "geduldiges Studium als Weg der Selbstvervollkommnung."
        ),
        "kernideen": [
            "Neo-konfuzianische Synthese",
            "Prinzip (Li) und Energie (Qi)",
            "Erkenntnis durch „Ergründen der Dinge“",
        ],
    },
    "personality-wang-yangming": {
        "beschreibung": (
            "Wang Yangming, General und Philosoph der Ming-Zeit, setzte dem "
            "gelehrten Studium Zhu Xis eine andere Betonung entgegen: Wissen "
            "und Handeln sind eine Einheit — wer wirklich weiß, handelt "
            "bereits danach. Im „angeborenen moralischen Wissen“ (Liangzhi) "
            "besitzt jeder Mensch einen unmittelbaren sittlichen Kompass."
        ),
        "kernideen": [
            "Einheit von Wissen und Handeln",
            "Liangzhi — angeborenes moralisches Wissen",
            "Intuition vor bloßer Gelehrsamkeit",
        ],
    },
    "personality-plotinus": {
        "beschreibung": (
            "Plotin gilt als Begründer des Neuplatonismus. In den "
            "„Enneaden“ entfaltet er eine gestufte Wirklichkeit: Aus dem "
            "„Einen“, das jenseits von Sein und Begriff liegt, geht der "
            "Geist (Nous) hervor, aus diesem die Seele, aus dieser die "
            "sinnliche Welt. Philosophie ist für ihn der Aufstieg der "
            "Seele, die Rückkehr zum Ursprung."
        ),
        "kernideen": [
            "Das „Eine“ jenseits von Sein und Begriff",
            "Stufung: Eines — Nous — Seele — Welt",
            "Philosophie als Aufstieg und Rückkehr",
        ],
    },
    "personality-porphyry": {
        "beschreibung": (
            "Porphyrios war Schüler Plotins und ordnete dessen verstreute "
            "Schriften zu den „Enneaden“ — ohne ihn wäre Plotins Werk kaum "
            "erhalten. Als eigenständiger Denker schrieb er zu Logik und "
            "Religionskritik; seine „Isagoge“ wurde für über ein "
            "Jahrtausend zum Standard-Lehrbuch der Logik."
        ),
        "kernideen": [
            "Herausgeber der „Enneaden“",
            "Die „Isagoge“ — Logik-Lehrbuch des Mittelalters",
            "Neuplatonismus systematisch vermittelt",
        ],
    },
    "personality-proclus": {
        "beschreibung": (
            "Proklos war der große Systematiker der spätantiken Akademie "
            "in Athen. In seiner „Theologischen Elementarlehre“ brachte er "
            "den Neuplatonismus in eine streng deduktive, fast "
            "mathematische Form: Satz für Satz entfaltet er, wie die Ebenen "
            "der Wirklichkeit auseinander hervorgehen. Sein System wurde "
            "Hauptquelle der mittelalterlichen Metaphysik."
        ),
        "kernideen": [
            "Deduktive Systematik der Metaphysik",
            "Logik der Vermittlung zwischen Seinsebenen",
            "Brücke in die mittelalterliche Philosophie",
        ],
    },
    "personality-descartes": {
        "beschreibung": (
            "René Descartes suchte ein Fundament des Wissens, das jedem "
            "Zweifel standhält. Methodisch zog er alles in Zweifel — bis er "
            "auf das eine Unbezweifelbare stieß: dass er, indem er zweifelt, "
            "denkt, und indem er denkt, ist. Mit dieser Verankerung der "
            "Gewissheit im denkenden Ich eröffnete er die neuzeitliche "
            "Philosophie."
        ),
        "kernideen": [
            "Methodischer Zweifel",
            "„Cogito ergo sum“",
            "Trennung von Geist und Körper",
        ],
    },
    "personality-kant": {
        "beschreibung": (
            "Immanuel Kant vollzog die „kopernikanische Wende“ der "
            "Philosophie: Nicht das Erkennen richtet sich nach den "
            "Gegenständen, sondern die Gegenstände erscheinen uns nach den "
            "Formen unseres Erkennens. In drei „Kritiken“ bestimmte er die "
            "Grenzen der Vernunft, den kategorischen Imperativ als Gesetz "
            "der Freiheit und die Urteilskraft."
        ),
        "kernideen": [
            "Kopernikanische Wende der Erkenntnis",
            "Der kategorische Imperativ",
            "Grenzen und Bedingungen der Vernunft",
        ],
    },
    "personality-hegel": {
        "beschreibung": (
            "G. W. F. Hegel entwarf das umfassendste System der klassischen "
            "deutschen Philosophie. Wirklichkeit ist für ihn nicht starr, "
            "sondern Prozess: Der „Geist“ entfaltet sich durch Widerspruch "
            "und dessen Aufhebung — die Dialektik — hindurch. Geschichte, "
            "Recht und Kunst erscheinen als Stufen, auf denen der Geist zu "
            "sich selbst kommt."
        ),
        "kernideen": [
            "Dialektik — Denken durch den Widerspruch",
            "Wirklichkeit als Prozess des Geistes",
            "Geschichte als Entfaltung der Freiheit",
        ],
    },
    "personality-nietzsche": {
        "beschreibung": (
            "Friedrich Nietzsche stellte die Herkunft der Moral selbst in "
            "Frage. Mit der „Genealogie“ deckte er auf, dass scheinbar "
            "ewige Werte geschichtlich geworden sind — und forderte ihre "
            "„Umwertung“. „Gott ist tot“ war für ihn die Diagnose eines "
            "Verlusts, der Antwort verlangt: Lebensbejahung noch im Leiden."
        ),
        "kernideen": [
            "Genealogie und Umwertung der Werte",
            "„Gott ist tot“ als kulturelle Diagnose",
            "Lebensbejahung trotz des Leidens",
        ],
    },
    "personality-camus": {
        "beschreibung": (
            "Albert Camus, Schriftsteller und Philosoph, dachte das "
            "„Absurde“ — den Zusammenstoß zwischen dem menschlichen "
            "Verlangen nach Sinn und einer schweigenden Welt. Seine Antwort "
            "ist nicht Resignation und nicht Selbsttäuschung, sondern "
            "Revolte: das hellsichtige, solidarische Festhalten am Leben "
            "ohne metaphysischen Trost."
        ),
        "kernideen": [
            "Das Absurde als Grundverhältnis",
            "Revolte statt Resignation oder Trost",
            "Solidarität ohne metaphysische Absicherung",
        ],
    },
}


def main() -> int:
    if not SHARED_CORE.exists():
        print(f"WARN: Quelle fehlt: {SHARED_CORE}", file=sys.stderr)
        if OUT.exists():
            print(f"OK: vorhandene Ausgabe bleibt erhalten: {OUT}")
            return 0
        print("FEHLER: keine Quelle und keine vorhandene Ausgabe.", file=sys.stderr)
        return 1

    src = json.loads(SHARED_CORE.read_text(encoding="utf-8"))
    personen = []
    missing = []
    for p in src:
        pid = p["id"]
        ext = EXPANDED.get(pid)
        if ext is None:
            missing.append(pid)
        personen.append({
            "id": pid,
            "name": p.get("name", ""),
            "years": p.get("years", ""),
            "role": p.get("role", ""),
            "tradition": p.get("tradition", ""),
            "symbol": p.get("symbol", ""),
            "accent": p.get("accent", "#b99b5d"),
            "wikiTerm": p.get("wikiTerm"),
            "portraitUrl": p.get("portraitUrl", ""),
            "source": p.get("source", ""),
            "relatedNodeIds": p.get("relatedNodeIds", []),
            "bio": p.get("bio", ""),                              # geprüfter Kern
            "beschreibung": (ext or {}).get("beschreibung", ""),  # modellgeneriert
            "kernideen": (ext or {}).get("kernideen", []),        # modellgeneriert
            "beschreibung_quelle": "model_generated" if ext else "none",
        })

    # Disziplinen aus den Traditionen ableiten (Reihenfolge wie Personen).
    ordered_trads = []
    for p in personen:
        if p["tradition"] and p["tradition"] not in ordered_trads:
            ordered_trads.append(p["tradition"])
    disziplinen = []
    for trad in ordered_trads:
        ids = [p["id"] for p in personen if p["tradition"] == trad]
        disziplinen.append({
            "id": trad.lower().replace(" ", "-"),
            "name": trad,
            "beschreibung": DISZIPLINEN.get(trad, ""),
            "beschreibung_quelle": "model_generated" if trad in DISZIPLINEN else "none",
            "person_ids": ids,
            "anzahl": len(ids),
        })

    out = {
        "schema_version": "1",
        "title": "TiSCH Personen-Bibliothek",
        "generated_from": "shared-core/data/personalities.json",
        "note": (
            "bio = geprüfter Kern aus dem shared-core (unverändert). "
            "beschreibung + kernideen + Disziplin-Texte sind modellgeneriert "
            "(beschreibung_quelle: model_generated) — ausführlicher "
            "Erläuterungstext, kein verifiziertes Wissen."
        ),
        "counts": {"personen": len(personen), "disziplinen": len(disziplinen)},
        "disziplinen": disziplinen,
        "personen": personen,
    }
    OUT.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OK: {OUT}")
    print(f"  {len(personen)} Personen, {len(disziplinen)} Disziplinen")
    if missing:
        print(f"  WARN: ohne erweiterte Beschreibung: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
