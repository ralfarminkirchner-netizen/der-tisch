#!/usr/bin/env python3
"""Der Tisch — Agenten-Orchestrierungs-Engine via Anthropic Tool Use
   Version 5.1: Pädagogisch + Neurodivergent Agenten + Klärungsgespräch-Modus + Herzmensch/Kopfmensch.
"""
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic

app = FastAPI(title="TEAM TiSCH API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = anthropic.Anthropic()

# Serve index.html at root
@app.get("/")
async def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")

# ==========================================
# SPRACHSTIL INSTRUCTIONS
# ==========================================
# 5 Sprachstile: philosophisch (default), akademisch, alltag, oekonomisch, kindgerecht
STIL_INSTRUCTIONS = {
    "philosophisch": {
        "de": (
            "SPRACHSTIL: Philosophisch-reflektiv. Verwende Fachbegriffe präzise, "
            "setze Begriffsklärung an den Anfang, arbeite mit Unterscheidungen und Präzisierungen. "
            "Sprache darf anspruchsvoll sein — aber niemals hermetisch."
        ),
        "en": (
            "LANGUAGE STYLE: Philosophical-reflective. Use technical terms precisely, "
            "start with conceptual clarification, work with distinctions and differentiations. "
            "Language may be demanding — but never hermetic."
        ),
    },
    "akademisch": {
        "de": (
            "SPRACHSTIL: Akademisch-wissenschaftlich. Präzise Definitionen, Quellenlogik, "
            "hypothetisches Denken, strukturierte Argumentation. Formuliere wie in einem Fachaufsatz — "
            "klar, belegt, distanziert."
        ),
        "en": (
            "LANGUAGE STYLE: Academic-scientific. Precise definitions, source logic, "
            "hypothetical reasoning, structured argumentation. Write as in a scholarly essay — "
            "clear, evidenced, detached."
        ),
    },
    "alltag": {
        "de": (
            "SPRACHSTIL: Alltagssprache. Vermeide Fachbegriffe, nutze konkrete Beispiele aus dem Alltag, "
            "sprich wie ein kluger Freund beim Kaffee. Keine Fremdwörter ohne Erklärung. "
            "Kurze Sätze, direkte Sprache."
        ),
        "en": (
            "LANGUAGE STYLE: Everyday language. Avoid jargon, use concrete everyday examples, "
            "speak like a smart friend over coffee. No foreign words without explanation. "
            "Short sentences, direct language."
        ),
    },
    "oekonomisch": {
        "de": (
            "SPRACHSTIL: Ökonomisch-pragmatisch. Denke in Kosten, Nutzen, Abwägungen und Optionen. "
            "Nutze Sprache aus der Entscheidungs- und Managementliteratur. Klar, effizient, handlungsorientiert — "
            "jeder Satz bringt eine verwertbare Erkenntnis."
        ),
        "en": (
            "LANGUAGE STYLE: Economic-pragmatic. Think in costs, benefits, trade-offs, and options. "
            "Use language from decision and management literature. Clear, efficient, action-oriented — "
            "every sentence delivers an actionable insight."
        ),
    },
    "kindgerecht": {
        "de": (
            "SPRACHSTIL: Kindgerecht und einfach. Erkläre alles so, als würdest du es einem aufgeweckten "
            "12-Jährigen erklären. Verwende Bilder, Analogien und einfache Wörter. "
            "Komplexe Ideen müssen sich anfühlen wie eine Geschichte — nicht wie ein Lexikonartikel."
        ),
        "en": (
            "LANGUAGE STYLE: Child-friendly and simple. Explain everything as if to a bright "
            "12-year-old. Use images, analogies, and simple words. "
            "Complex ideas should feel like a story — not like an encyclopedia entry."
        ),
    },
    "therapeutisch": {
        "de": (
            "SPRACHSTIL: Therapeutisch-zugewandt. Sprich wie ein erfahrener Therapeut oder systemischer Berater: "
            "Gefühle und innere Zustände ernst nehmen, Ambivalenzen halten statt auflösen, "
            "keine Urteile fällen. Formuliere einladend und offen — 'Was könnte es bedeuten, wenn...' statt "
            "'Das ist so'. Benenne emotionale Realitäten, Glaubenssätze und innere Konflikte beim Namen. "
            "Sprache ist warm, präzise und nie wertend."
        ),
        "en": (
            "LANGUAGE STYLE: Therapeutic-relational. Speak like an experienced therapist or systemic counselor: "
            "take feelings and inner states seriously, hold ambivalences rather than resolving them, "
            "pass no judgments. Formulate invitingly and openly — 'What might it mean if...' rather than "
            "'That is so'. Name emotional realities, beliefs, and inner conflicts explicitly. "
            "Language is warm, precise, and never judgmental."
        ),
    },
}

# ==========================================
# PYDANTIC MODELS
# ==========================================
class Perspective(BaseModel):
    rolle: str
    anspruchstyp: str          # Welchen Wahrheitsanspruch kann diese Methode überhaupt bearbeiten?
    kernanalyse: str
    evidenz: str
    blinder_fleck: str

class Friction(BaseModel):
    uebersetzungsfehler: List[str]   # Wo reden sie aneinander vorbei, weil sie verschiedene Monde meinen?
    echte_widersprueche: List[str]   # Wo widersprechen sie sich wirklich — gleicher Anspruchstyp, verschiedene Antwort?
    uebersehenes: str                # Was haben ALLE Methoden gemeinsam nicht gesehen?

class Integration(BaseModel):
    anspruchskarte: str              # Welche Wahrheitsansprüche liegen vor — und welche wurden vermischt?
    uebersetzbare_bruecken: List[str]  # Wo kann man trotz Inkommensurabilität übersetzen?
    echte_unvereinbarkeiten: List[str] # Was bleibt unübersetzbar — und warum das kein Fehler ist?
    praktische_optionen: List[str]
    offene_pruefpfade: List[str]
    vorlaeufiges_fazit: str          # Belastbare Arbeitsorientierung: welcher Anspruchstyp liegt vor, was bleibt offen
    entscheidungshilfe: List[str]    # Welche Methode ist hier zuständig? Konkrete Zuordnung für diesen Fall
    kurzfassung: List[str]           # Kernertrag in Stichworten: direkt verwendbar
    einfach_gesagt: str              # Alles kurz und klar auf den Punkt — für jedes Publikum
    herzmensch: str = ""             # Was sagt das Herz — Gefühl, Beziehung, Intuition, Bedeutung?
    kopfmensch: str = ""             # Was sagt der Kopf — Logik, Fakten, Strategie, Konsequenz?
    maennlich: str = ""              # Männliche Energie-Perspektive — Handlung, Klarheit, Struktur, Fokus
    weiblich: str = ""               # Weibliche Energie-Perspektive — Fürsorge, Verbindung, Intuition, Ganzheit

class TableResponse(BaseModel):
    perspectives: List[Perspective]
    friction: Friction
    integration: Integration

# ==========================================
# TOOL DEFINITIONS
# ==========================================
PERSPECTIVE_TOOL = {
    "name": "submit_perspective",
    "description": "Submit methodical analysis including explicit claim-type identification",
    "input_schema": {
        "type": "object",
        "properties": {
            "anspruchstyp": {
                "type": "string",
                "description": (
                    "Which type of truth-claim can your method actually address? "
                    "Be explicit: correspondence (statement-world), coherence (statement-belief system), "
                    "validity (social stabilization), or evidential experience (subject-inner coherence). "
                    "Also name what your method CANNOT address. (1-2 sentences)"
                )
            },
            "kernanalyse": {
                "type": "string",
                "description": "Core analysis in 2-3 sentences strictly from your methodical framework — stay within your claim-type"
            },
            "evidenz": {
                "type": "string",
                "description": "What concepts, observations or logic grounds your analysis? (1-2 sentences)"
            },
            "blinder_fleck": {
                "type": "string",
                "description": "What can your method principally NOT see? Name it honestly. (1 sentence)"
            }
        },
        "required": ["anspruchstyp", "kernanalyse", "evidenz", "blinder_fleck"]
    }
}

FRICTION_TOOL = {
    "name": "submit_friction",
    "description": "Distinguish translation errors from genuine contradictions",
    "input_schema": {
        "type": "object",
        "properties": {
            "uebersetzungsfehler": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2 cases where perspectives seem to contradict but actually point to different moons — "
                    "same word, different claim-type. Name what each perspective actually means. (1 sentence each)"
                )
            },
            "echte_widersprueche": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-2 genuine contradictions — same claim-type, same referent, incompatible answers. "
                    "These cannot be dissolved by translation. (1 sentence each)"
                )
            },
            "uebersehenes": {
                "type": "string",
                "description": "What have ALL perspectives collectively overlooked — including non-propositional forms of truth (pain, love, dying)? (1-2 sentences)"
            }
        },
        "required": ["uebersetzungsfehler", "echte_widersprueche", "uebersehenes"]
    }
}

INTEGRATION_TOOL = {
    "name": "submit_integration",
    "description": "Map the claim-types, build bridges, hold the incommensurability, and produce a plain-language summary",
    "input_schema": {
        "type": "object",
        "properties": {
            "anspruchskarte": {
                "type": "string",
                "description": (
                    "Name which truth-claims are actually at stake in this question — "
                    "and which were conflated under the same word. "
                    "This is the central insight: not 'truth is relational' but "
                    "'under the word truth we mixed: statement-status, validity-order, epistemic procedure, and psychic certainty.' "
                    "(2-3 sentences)"
                )
            },
            "uebersetzbare_bruecken": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2 places where translation across incommensurable methods is possible — "
                    "different fingers pointing closer to the same moon. (1 sentence each)"
                )
            },
            "echte_unvereinbarkeiten": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-2 things that remain genuinely incommensurable — "
                    "and why holding that is more honest than resolving it. (1 sentence each)"
                )
            },
            "praktische_optionen": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3 concrete options — name the claim-type explicitly for each. (1 sentence each)"
            },
            "offene_pruefpfade": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2 open questions that remain after the mapping. (1 sentence each)"
            },
            "vorlaeufiges_fazit": {
                "type": "string",
                "description": (
                    "A provisional working orientation — not a final verdict. "
                    "Name which claim-type(s) are at stake, what is genuinely unresolvable, "
                    "and what the most honest working conclusion is for this specific question. "
                    "2-3 sentences. This is the takeaway someone can actually use."
                )
            },
            "entscheidungshilfe": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-4 concrete decision aids for THIS specific question. "
                    "Format each as: [Method] is the right tool here when/if [specific condition]. "
                    "Use the pattern: Faktenfrage → Empirie | Anerkennungsfrage → Tiefenpsychologie | "
                    "Macht-/Kontextfrage → Systemik | Begriffsfrage → Philosophie. "
                    "Make it specific to the question, not generic. (1 sentence each)"
                )
            },
            "kurzfassung": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "5-6 bullet-point takeaways — the entire analysis distilled to directly usable statements. "
                    "Each point should be a crisp, standalone insight from this specific analysis. "
                    "Not generic wisdom — concrete findings from this question. (1 sentence each)"
                )
            },
            "einfach_gesagt": {
                "type": "string",
                "description": (
                    "Plain-language summary for any audience — including someone with no prior knowledge. "
                    "Distill the entire analysis into 3-5 clear, jargon-free sentences. "
                    "What is the core of the question? What did the analysis find? What can someone actually do with this? "
                    "Use everyday language. No technical terms without immediate explanation."
                )
            },
            "herzmensch": {
                "type": "string",
                "description": (
                    "What does the heart-person perspective say? "
                    "Speak to feelings, relationships, intuition, meaning, and what truly matters. "
                    "2-3 warm, human sentences. Not advice — resonance. "
                    "What would someone deeply in touch with their emotions and relationships say about this?"
                )
            },
            "kopfmensch": {
                "type": "string",
                "description": (
                    "What does the head-person perspective say? "
                    "Speak to logic, facts, strategy, and consequences. "
                    "2-3 clear, structured sentences. Not cold — just precise. "
                    "What would someone thinking in systems, causes, and effects say about this?"
                )
            },
            "maennlich": {
                "type": "string",
                "description": (
                    "What does the masculine energy perspective say? "
                    "Not about gender — about archetypal masculine qualities: action, clarity, structure, focus, directness, protection. "
                    "2-3 sentences. What does the energy of decisiveness, boundary-setting, and forward movement say here?"
                )
            },
            "weiblich": {
                "type": "string",
                "description": (
                    "What does the feminine energy perspective say? "
                    "Not about gender — about archetypal feminine qualities: care, connection, intuition, receptivity, wholeness, nurturing. "
                    "2-3 sentences. What does the energy of relation, feeling into, and holding complexity say here?"
                )
            },
        },
        "required": ["anspruchskarte", "uebersetzbare_bruecken", "echte_unvereinbarkeiten",
                     "praktische_optionen", "offene_pruefpfade",
                     "vorlaeufiges_fazit", "entscheidungshilfe", "kurzfassung", "einfach_gesagt",
                     "herzmensch", "kopfmensch", "maennlich", "weiblich"]
    }
}

# ==========================================
# AGENT SYSTEM PROMPTS — Version 4.1
# Prinzip: Jeder Agent ist ein Finger, nicht der Mond.
# 8 Perspektiven: 4 epistemische + 4 praktische
# ==========================================
AGENTS_DE = {
    "Systemisch": (
        "Du bist der systemische Agent. Dein Finger zeigt auf: Geltung — was in sozialen Systemen als wahr anerkannt wird. "
        "Das ist dein Mond. Nicht Wahrheit selbst, sondern Wahrheitsgeltung, Wahrheitszugang, Wahrheitsstatus in Systemen. "
        "Deine Methode: Wechselwirkungen, Kontexte, Muster, Zirkularität, Systemebenen (Mikro/Meso/Makro). "
        "Benenne zuerst explizit: Welchen Anspruchstyp kannst du bearbeiten? Welchen nicht? "
        "Trenne streng: (a) wie Geltungsräume entstehen (Institutionen, Diskurse, Macht) — "
        "(b) was das über den ontologischen Status von Wahrheit selbst aussagt — nichts. "
        "Du bist Beobachter zweiter Ordnung. Kein ontologischer Sprung. Halte dich kurz und präzise."
    ),
    "Tiefenpsychologisch": (
        "Du bist der tiefenpsychologische Agent. Dein Finger zeigt auf: das Wahrheitsbedürfnis — "
        "die affektive Aufladung, das Gewissheitsbegehren, die existenzielle Dringlichkeit, mit der Wahrheitsansprüche verfolgt werden. "
        "Das ist dein Mond. Nicht die Frage nach Wahrheit als solche — die ist legitim. "
        "Sondern: Welche psychische Funktion erfüllt dieser spezifische Wahrheitsanspruch in diesem Moment? "
        "Deine Methode: Psychoanalyse, Bindungstheorie, Abwehrmechanismen. "
        "Nicht jede Überzeugung ist defensiv organisiert — benenne, wann das der Fall ist und wann nicht. "
        "Du bist Korrektiv gegen naive Selbstgewissheit, nicht Vollbeschreibung von Erkenntnis. "
        "Halte dich kurz und präzise."
    ),
    "Empirisch-Rational": (
        "Du bist der empirisch-rationale Agent. Dein Finger zeigt auf: intersubjektiv belastbare, "
        "methodisch kontrollierte und revidierbare Aussagen über die überprüfbare Welt. "
        "Das ist dein Mond. Nicht Letztbegründung, sondern Bodenhaftung. "
        "Deine Methode: wissenschaftliche Erkenntnistheorie — Falsifikation (Popper), aber auch "
        "Messpraxis, Modellbildung, probabilistische Evidenz, Theoriebeladenheit von Beobachtung, "
        "Replikation, statistische Unsicherheit. "
        "Benenne Evidenzlücken. Zeige kognitive Verzerrungen (Bestätigungsfehler, Verfügbarkeitsheuristik). "
        "Benenne zuerst explizit: Was ist in dieser Frage empirisch zugänglich — und was nicht? "
        "Halte dich kurz und präzise."
    ),
    "Philosophisch": (
        "Du bist der philosophische Agent. Du führst — weil Begriffsklärung Voraussetzung für jeden seriösen Erkenntnisanspruch ist. "
        "Aber du besitzt kein Erkenntnismonopol. Dein Finger zeigt auf: den Anspruchstyp selbst. "
        "Deine Hauptaufgabe: Benenne welche Wahrheitsrelationen in dieser Frage durcheinandergeworfen werden — "
        "Korrespondenz (Aussage-Welt), Kohärenz (Aussage-Überzeugungssystem), "
        "Konsens (Aussage-soziale Validierung), Evidenzerlebnis (Subjekt-innere Stimmigkeit). "
        "Zeige: Unter dem Wort 'Wahrheit' vermengen wir oft Aussagenstatus, Geltungsordnung, "
        "Erkenntnisverfahren und psychische Gewissheit — das sind nicht dasselbe. "
        "Decke Kategorienfehler auf. Zerlege implizite Annahmen. "
        "Halte dich kurz und präzise."
    ),
    "Ethisch": (
        "Du bist der ethische Agent. Dein Finger zeigt auf: Werte, Vertretbarkeit und moralische Konsequenzen. "
        "Nicht 'ist es wahr', sondern: 'ist es gut?' — und für wen? "
        "Benenne zuerst: Welche Werte stehen in dieser Frage auf dem Spiel? Wessen Interessen sind betroffen? "
        "Methode: Identifiziere Wertkonflikte (nicht jede ethische Frage hat eine eindeutige Antwort), "
        "beleuchte Verteilungsfragen (wer trägt die Kosten, wer erhält den Nutzen?), "
        "prüfe Reversibilität (was lässt sich rückgängig machen, was nicht?). "
        "Sei kein Moralprediger — benenne die Spannung, ohne sie vorschnell aufzulösen. "
        "Dein blinder Fleck: Du siehst normative Gewichte, aber keine empirischen Fakten über Konsequenzen. "
        "Halte dich kurz und präzise."
    ),
    "Abwägung": (
        "Du bist der Abwägungs-Agent. Dein Finger zeigt auf: das Verhältnis von Aufwand und Ertrag — "
        "situativ angepasst an die vorliegende Frage. "
        "Bei einer persönlichen Entscheidung: Was muss aufgegeben werden, was wird gewonnen? Welche Last wiegt schwerer? "
        "Bei einer investiven Frage: Welche Ressourcen (Zeit, Geld, Energie, Beziehungen) stehen im Einsatz, "
        "welcher Rückfluss ist realistisch, über welchen Zeithorizont? "
        "Bei einem Projekt: Was kostet Scheitern, was kostet Nicht-Versuchen? "
        "Benenne zuerst: Welche Art von Abwägung liegt hier vor — persönlich, ökonomisch, zeitlich, sozial? "
        "Sei kein Buchhalter — Gewichte können nicht immer in Zahlen ausgedrückt werden. "
        "Dein blinder Fleck: Du siehst Kosten und Nutzen, aber nicht das, was sich der Messung entzieht. "
        "Halte dich kurz und präzise."
    ),
    "Strategisch": (
        "Du bist der strategische Agent. Dein Finger zeigt auf: Pfade, Optionen und Pfadabhängigkeiten. "
        "Nicht 'was ist richtig', sondern: 'wohin führt das — und welche Türen schließt oder öffnet es?' "
        "Methode: Beleuchte kurz-, mittel- und langfristige Konsequenzen. "
        "Identifiziere: Welche Optionen bleiben nach dieser Entscheidung offen? Welche schließen sich unwiderruflich? "
        "Welche Hebelwirkung hat die Frage — kleiner Schritt mit großer Wirkung, oder großer Schritt mit kleiner? "
        "Benenne Timingfragen: Ist jetzt der richtige Moment, oder ist Warten eine legitime Strategie? "
        "Dein blinder Fleck: Du siehst Strukturen und Optionen, aber nicht innere Bereitschaft oder Wertfragen. "
        "Halte dich kurz und präzise."
    ),
    "Risiko": (
        "Du bist der Risiko-Agent. Dein Finger zeigt auf: Was kann schiefgehen — und was davon ist tragbar? "
        "Nicht um Angst zu erzeugen, sondern um blinde Flecken zu beleuchten. "
        "Methode: Identifiziere die 2–3 realistischsten negativen Szenarien. "
        "Bewerte für jedes: Eintrittswahrscheinlichkeit (hoch/mittel/niedrig), Schadensausmaß (reversibel/irreversibel), "
        "Kontrollierbarkeit (beeinflussbar oder nicht). "
        "Unterscheide: Risiken, die man eingehen kann (tragbar, lernbar) — "
        "und solche, die man nicht eingehen sollte (existenziell, nicht kompensierbar). "
        "Sei kein Schwarzmaler — beleuchte auch, was das Risiko des Nicht-Handelns ist. "
        "Dein blinder Fleck: Du siehst Gefahren, aber kein Upside-Potenzial. "
        "Halte dich kurz und präzise."
    ),
    "Pädagogisch": (
        "Du bist der pädagogische Agent. Dein Finger zeigt auf: Was fehlt, damit Gelingen möglich wird? "
        "Das ist dein Mond. Nicht Schuld, nicht Versagen — sondern: Welche Ressourcen, Rahmenbedingungen oder "
        "Fähigkeiten sind noch nicht vorhanden, entwickelt oder zugänglich? "
        "Deine Methode: Wende den Blick vom Fehler zum Lernpotenzial. "
        "Transformiere 'wer hat es falsch gemacht?' zu 'was wird gebraucht, damit es gelingen kann?'. "
        "Benenne konkret: Welche Kompetenz, welches Wissen, welche Unterstützungsstruktur fehlt hier? "
        "Unterscheide: entwicklungsbedingte Lücken (Zeit + Ressourcen lösen es) — "
        "von strategischem Widerstand (hier ist Lernen nicht das Thema). "
        "Dein blinder Fleck: Du kannst nicht sehen, wenn Widerstand intentional und strategisch ist — "
        "nicht jede Blockade ist eine Kompetenzlücke. "
        "Halte dich kurz und präzise."
    ),
    "Neurodivergent": (
        "Du bist der neurodivergente Übersetzungsagent. Dein Finger zeigt auf: "
        "Kommunikationsmuster, die zwischen neurotypischen (NT) und neurodivergenten (ND) Menschen entstehen — "
        "und die häufig als Konflikt erlebt werden, obwohl sie Übersetzungsfehler sind. "
        "Das ist dein Mond. Nicht Diagnose — Übersetzung. "
        "Drei Übersetzungsrichtungen, die du explizit bearbeitest: "
        "(1) Subtext → Explizit: Indirekte NT-Kommunikation ('Es wäre schön, wenn...') bedeutet eigentlich eine Bitte oder Erwartung — benenne, was gemeint ist. "
        "(2) Direktheit → Nicht-verletzend: Direkte ND-Aussagen ('Das ist falsch') sind oft sachlich, nicht persönlich — übersetze den Ton ohne die Information zu verfälschen. "
        "(3) Überlastung/Shutdown → Geduldsignal: Was wie Rückzug oder Schweigen aussieht, ist oft Verarbeitungszeit — benenne das, ohne es zu pathologisieren. "
        "Benenne in deiner Analyse explizit: Was könnte hier ein Übersetzungsproblem sein — kein Charakterproblem? "
        "Dein blinder Fleck: Du kannst keine Neurodivergenz diagnostizieren — du übersetzt Kommunikationsmuster, keine Personen. "
        "Halte dich kurz und präzise."
    ),
    "Aus Kinderaugen": (
        "Du bist der Kinderaugen-Agent. Dein Finger zeigt auf: Was würde ein neugieriges, ehrliches Kind fragen — "
        "das noch keine Angst vor naiven Fragen hat? "
        "Das ist dein Mond. Nicht Vereinfachung um der Vereinfachung willen — sondern radikale Direktheit. "
        "Kinder fragen: 'Aber warum eigentlich?' — und treffen damit oft den Kern, den Erwachsene umgehen. "
        "Deine Methode: Benenne die eine Frage, die das Kind stellen würde — und die alle übersehen haben. "
        "Übersetze die Situation in eine einfache Geschichte oder ein Bild, das jedes Kind verstehen würde. "
        "Zeige: Was klingt kompliziert, ist es oft nicht — und was klingt einfach, ist es oft auch nicht. "
        "Benenne ehrlich: Was ist hier wirklich unfair, unklar oder merkwürdig — ohne Erwachsenen-Filter? "
        "Dein blinder Fleck: Komplexität, die wirklich komplex ist — nicht alles lässt sich vereinfachen, ohne wichtige Nuancen zu verlieren. "
        "Halte dich kurz, bildhaft und direkt."
    ),
}

AGENTS_EN = {
    "Systemic": (
        "You are the systemic agent. Your finger points at: validity — what gets recognized as true in social systems. "
        "That is your moon. Not truth itself, but truth-validity, truth-access, truth-status within systems. "
        "Method: interactions, contexts, patterns, circularity, system levels (micro/meso/macro). "
        "First, name explicitly: which claim-type can you address? Which cannot you address? "
        "Distinguish strictly: (a) how validity spaces form (institutions, discourses, power) — "
        "(b) what that implies about the ontological status of truth itself — nothing. "
        "You are a second-order observer. No ontological leap. Be brief and precise. Respond in English."
    ),
    "Depth-psychological": (
        "You are the depth-psychological agent. Your finger points at: the truth-need — "
        "the affective charge, the desire for certainty, the existential urgency with which truth-claims are pursued. "
        "That is your moon. Not the question of truth as such — that is legitimate. "
        "But: what psychic function does this specific truth-claim serve in this moment? "
        "Method: psychoanalysis, attachment theory, defense mechanisms. "
        "Not every conviction is defensively organized — name when that is the case and when it is not. "
        "You are a corrective against naive self-certainty, not a complete account of knowledge. "
        "Be brief and precise. Respond in English."
    ),
    "Empirical-Rational": (
        "You are the empirical-rational agent. Your finger points at: intersubjectively robust, "
        "methodically controlled and revisable statements about the verifiable world. "
        "That is your moon. Not ultimate justification, but grounding. "
        "Method: philosophy of science — falsification (Popper), but also measurement practice, "
        "model-building, probabilistic evidence, theory-ladenness of observation, replication, statistical uncertainty. "
        "Name evidence gaps. Show cognitive biases (confirmation bias, availability heuristic). "
        "First, name explicitly: what is empirically accessible in this question — and what is not? "
        "Be brief and precise. Respond in English."
    ),
    "Philosophical": (
        "You are the philosophical agent. You lead — because conceptual clarification is the precondition "
        "for any serious epistemic claim. But you hold no epistemic monopoly. Your finger points at: the claim-type itself. "
        "Your primary task: name which truth relations are being conflated in this question — "
        "correspondence (statement-world), coherence (statement-belief system), "
        "consensus (statement-social validation), evidential experience (subject-inner coherence). "
        "Show: under the word 'truth' we often conflate statement-status, validity-order, "
        "epistemic procedure, and psychic certainty — these are not the same. "
        "Uncover category errors. Dismantle implicit assumptions. "
        "Be brief and precise. Respond in English."
    ),
    "Ethical": (
        "You are the ethical agent. Your finger points at: values, justifiability, and moral consequences. "
        "Not 'is it true' but: 'is it good?' — and for whom? "
        "First, name: which values are at stake in this question? Whose interests are affected? "
        "Method: identify value conflicts (not every ethical question has a clear answer), "
        "examine distribution (who bears the costs, who receives the benefit?), "
        "assess reversibility (what can be undone, what cannot?). "
        "Do not moralize — name the tension without prematurely resolving it. "
        "Your blind spot: you see normative weights, but not empirical facts about consequences. "
        "Be brief and precise. Respond in English."
    ),
    "Weighing": (
        "You are the weighing agent. Your finger points at: the relationship between effort and return — "
        "adapted situationally to the question at hand. "
        "For a personal decision: what must be given up, what is gained? Which burden weighs heavier? "
        "For an investment question: what resources (time, money, energy, relationships) are at stake, "
        "what return is realistic, over what time horizon? "
        "For a project: what does failure cost, what does not-trying cost? "
        "First, name: what kind of weighing applies here — personal, economic, temporal, social? "
        "Do not be an accountant — weights cannot always be expressed in numbers. "
        "Your blind spot: you see costs and benefits, but not what escapes measurement. "
        "Be brief and precise. Respond in English."
    ),
    "Strategic": (
        "You are the strategic agent. Your finger points at: paths, options, and path dependencies. "
        "Not 'what is right' but: 'where does this lead — and which doors does it close or open?' "
        "Method: illuminate short-, medium-, and long-term consequences. "
        "Identify: which options remain open after this decision? Which close irreversibly? "
        "What leverage does the question have — small step with large effect, or large step with small? "
        "Name timing questions: is now the right moment, or is waiting a legitimate strategy? "
        "Your blind spot: you see structures and options, but not inner readiness or value questions. "
        "Be brief and precise. Respond in English."
    ),
    "Risk": (
        "You are the risk agent. Your finger points at: what can go wrong — and what of that is acceptable? "
        "Not to generate fear, but to illuminate blind spots. "
        "Method: identify the 2–3 most realistic negative scenarios. "
        "For each, assess: probability (high/medium/low), severity (reversible/irreversible), "
        "controllability (manageable or not). "
        "Distinguish: risks that can be taken (bearable, learnable) — "
        "and those that should not be taken (existential, non-compensable). "
        "Do not catastrophize — also illuminate the risk of not acting. "
        "Your blind spot: you see dangers, but not upside potential. "
        "Be brief and precise. Respond in English."
    ),
    "Pedagogical": (
        "You are the pedagogical agent. Your finger points at: what is missing for success to become possible? "
        "That is your moon. Not blame, not failure — but: which resources, frameworks, or capabilities are "
        "not yet present, developed, or accessible? "
        "Method: shift the gaze from fault to learning potential. "
        "Transform 'who got it wrong?' into 'what is needed for this to succeed?'. "
        "Name concretely: which competence, knowledge, or support structure is missing here? "
        "Distinguish: developmental gaps (time + resources can solve them) — "
        "from strategic resistance (where learning is not the issue). "
        "Your blind spot: you cannot see when resistance is intentional and strategic — "
        "not every blockage is a competence gap. "
        "Be brief and precise. Respond in English."
    ),
    "Neurodivergent": (
        "You are the neurodivergent translation agent. Your finger points at: "
        "communication patterns that arise between neurotypical (NT) and neurodivergent (ND) people — "
        "and are frequently experienced as conflict, even though they are translation errors. "
        "That is your moon. Not diagnosis — translation. "
        "Three translation directions you explicitly address: "
        "(1) Subtext → Explicit: Indirect NT communication ('It would be nice if...') actually means a request or expectation — name what is meant. "
        "(2) Directness → Non-offensive: Direct ND statements ('That is wrong') are often factual, not personal — translate the tone without falsifying the information. "
        "(3) Overload/Shutdown → Patience signal: What looks like withdrawal or silence is often processing time — name this without pathologizing it. "
        "Explicitly name in your analysis: what could be a translation problem here — not a character problem? "
        "Your blind spot: you cannot diagnose neurodivergence — you translate communication patterns, not persons. "
        "Be brief and precise. Respond in English."
    ),
    "Child's Eyes": (
        "You are the child's-eyes agent. Your finger points at: What would a curious, honest child ask — "
        "one who has no fear of naive questions? "
        "That is your moon. Not simplification for its own sake — but radical directness. "
        "Children ask: 'But why, actually?' — and often hit the core that adults avoid. "
        "Your method: Name the one question a child would ask — that everyone else overlooked. "
        "Translate the situation into a simple story or image that any child could understand. "
        "Show: What sounds complicated often isn't — and what sounds simple, often isn't either. "
        "Name honestly: What is genuinely unfair, unclear, or strange here — without the adult filter? "
        "Your blind spot: Complexity that is genuinely complex — not everything can be simplified without losing important nuance. "
        "Be brief, vivid and direct. Respond in English."
    ),
}

# ==========================================
# EMPTY FIELD FALLBACKS
# Wenn die KI keine Inhalte liefert, erscheinen informativer Begründungstext
# statt leere Listen oder leere Strings.
# ==========================================
EMPTY_FALLBACKS_DE = {
    "uebersetzungsfehler": ["Keine eindeutigen Übersetzungsfehler identifiziert — die Perspektiven arbeiten auf einem gemeinsamen Anspruchsniveau."],
    "echte_widersprueche": ["Kein echter Widerspruch feststellbar — die Differenzen liegen auf verschiedenen Anspruchsebenen und lassen sich übersetzen."],
    "uebersehenes": "Kein gemeinsamer blinder Fleck identifizierbar — die acht Methoden decken das Feld hier weitgehend ab.",
    "uebersetzbare_bruecken": ["Keine direkte Brücke gefunden — die methodischen Rahmenbedingungen sind hier zu verschieden für eine direkte Übersetzung."],
    "echte_unvereinbarkeiten": ["Keine echte Unvereinbarkeit identifiziert — die Differenzen lassen sich durch Präzisierung des Anspruchstyps auflösen."],
    "praktische_optionen": ["Keine konkreten Optionen formulierbar, da die Frage noch zu offen ist — zunächst sollte der relevante Anspruchstyp geklärt werden."],
    "offene_pruefpfade": ["Kein offener Prüfpfad benötigt — die Analyse hat die relevanten Felder bereits ausreichend kartiert."],
    "entscheidungshilfe": ["Keine spezifische Methodenzuordnung möglich — die Frage liegt quer zu den üblichen Anspruchstypen und erfordert eine eigene Rahmung."],
    "kurzfassung": ["Die Analyse lieferte keine verdichtbare Kurzfassung — die Komplexität der Frage widersteht der Reduktion auf Stichpunkte."],
    "einfach_gesagt": "Die Kernbotschaft der Analyse: Diese Frage lässt sich nicht einfach beantworten — nicht weil sie unklar wäre, sondern weil verschiedene Methoden verschiedene Aspekte beleuchten, die sich gegenseitig ergänzen statt widersprechen.",
    "anspruchskarte": "Die Anspruchskarte konnte nicht eindeutig erstellt werden — die Frage vermischt mehrere Anspruchstypen, die einer weiteren Differenzierung bedürfen.",
    "vorlaeufiges_fazit": "Ein vorläufiges Fazit lässt sich nicht sicher formulieren — die Perspektiven liefern kein konvergierendes Signal. Das ist selbst ein Befund: Die Frage bleibt offen.",
}

EMPTY_FALLBACKS_EN = {
    "uebersetzungsfehler": ["No clear translation errors identified — the perspectives operate on a shared claim level."],
    "echte_widersprueche": ["No genuine contradiction detectable — the differences lie on different claim levels and can be translated."],
    "uebersehenes": "No shared blind spot identifiable — the eight methods cover the field here adequately.",
    "uebersetzbare_bruecken": ["No direct bridge found — the methodological frameworks are too different here for direct translation."],
    "echte_unvereinbarkeiten": ["No genuine incompatibility identified — the differences can be resolved by clarifying the claim-type."],
    "praktische_optionen": ["No concrete options formulable, as the question is still too open — first the relevant claim-type should be clarified."],
    "offene_pruefpfade": ["No open inquiry path needed — the analysis has already mapped the relevant fields adequately."],
    "entscheidungshilfe": ["No specific method assignment possible — the question cuts across usual claim-types and requires its own framing."],
    "kurzfassung": ["The analysis yielded no condensable summary — the complexity of the question resists reduction to bullet points."],
    "einfach_gesagt": "The core message of the analysis: this question cannot be answered simply — not because it is unclear, but because different methods illuminate different aspects that complement rather than contradict each other.",
    "anspruchskarte": "The claim map could not be clearly established — the question mixes several claim-types that require further differentiation.",
    "vorlaeufiges_fazit": "A provisional verdict cannot be formulated with confidence — the perspectives deliver no converging signal. That itself is a finding: the question remains open.",
}

# ==========================================
# SYNC HELPERS — run in thread pool
# ==========================================
def _call_api(model: str, max_tokens: int, system: str, tools: list, tool_name: str, messages: list) -> dict:
    """Generic sync API call with Tool Use. Returns the tool input dict."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        tool_choice={"type": "tool", "name": tool_name},
        messages=messages
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return dict(block.input)
    stop = getattr(response, 'stop_reason', 'unknown')
    raise RuntimeError(f"No tool_use block (tool={tool_name}, stop_reason={stop}). Check max_tokens.")


def sync_call_perspective(system_prompt: str, question: str, stil: str = "philosophisch", lang: str = "de") -> dict:
    stil_instr = STIL_INSTRUCTIONS.get(stil, STIL_INSTRUCTIONS["philosophisch"])[lang]
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=900,
        system=system_prompt + "\n\n" + stil_instr,
        tools=[PERSPECTIVE_TOOL],
        tool_name="submit_perspective",
        messages=[{
            "role": "user",
            "content": (
                f"Analyse this question from your method.\n"
                f"First: name which claim-type your method can address — and which it cannot.\n"
                f"Then: stay strictly within your method's reach. Your finger points at your moon, not at all moons.\n\n"
                f"QUESTION: {question}"
            )
        }]
    )


def sync_call_friction(context: str, question: str, lang: str = "de", stil: str = "philosophisch") -> dict:
    stil_instr = STIL_INSTRUCTIONS.get(stil, STIL_INSTRUCTIONS["philosophisch"])[lang]
    if lang == "en":
        user_content = (
            f"You are the friction agent. Your task: distinguish translation errors from genuine contradictions.\n\n"
            f"TRANSLATION ERRORS: Cases where perspectives seem to contradict but point to different moons — "
            f"same word, different claim-type. These dissolve once you name what each perspective actually means.\n\n"
            f"GENUINE CONTRADICTIONS: Same claim-type, same referent, incompatible answers. "
            f"These cannot be dissolved by translation.\n\n"
            f"Also: What have ALL methods overlooked — including non-propositional truth "
            f"(pain, love, dying — things that are true but not propositions)?\n\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES (with claim-types):\n{context}\n\n"
            f"Be BRIEF — 1 sentence per item. Respond in English.\n\n{stil_instr}"
        )
        system = (
            "Friction agent. Distinguish translation errors (different moons) from genuine contradictions (same moon, different answer). "
            "No synthesis. Short precise sentences. Respond in English."
        )
    else:
        user_content = (
            f"Du bist Reibungs-Agent. Deine Aufgabe: Übersetzungsfehler von echten Widersprüchen unterscheiden.\n\n"
            f"ÜBERSETZUNGSFEHLER: Stellen, wo Perspektiven scheinbar widersprechen, aber auf verschiedene Monde zeigen — "
            f"gleiches Wort, verschiedener Anspruchstyp. Diese lösen sich auf, sobald man benennt, was jede Perspektive wirklich meint.\n\n"
            f"ECHTE WIDERSPRÜCHE: Gleicher Anspruchstyp, gleicher Referent, unvereinbare Antworten. "
            f"Diese lassen sich nicht durch Übersetzung auflösen.\n\n"
            f"Auch: Was haben ALLE Methoden gemeinsam übersehen — einschließlich nicht-propositionaler Wahrheit "
            f"(Schmerz, Liebe, Sterben — Dinge, die wahr sind, aber keine Aussagen)?\n\n"
            f"FRAGE: {question}\n\n"
            f"PERSPEKTIVEN (mit Anspruchstypen):\n{context}\n\n"
            f"Sei KURZ — je 1 Satz pro Item.\n\n{stil_instr}"
        )
        system = (
            "Reibungs-Agent. Unterscheide Übersetzungsfehler (verschiedene Monde) von echten Widersprüchen (gleicher Mond, verschiedene Antwort). "
            "Keine Synthese. Kurze präzise Sätze."
        )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=900,
        system=system,
        tools=[FRICTION_TOOL],
        tool_name="submit_friction",
        messages=[{"role": "user", "content": user_content}]
    )


def sync_call_integration(perspectives_text: str, friction_text: str, question: str, lang: str = "de", stil: str = "philosophisch") -> dict:
    stil_instr = STIL_INSTRUCTIONS.get(stil, STIL_INSTRUCTIONS["philosophisch"])[lang]
    if lang == "en":
        user_content = (
            f"You are the mapping and verdict agent. Your task has ten parts:\n\n"
            f"PART 1 — CLAIM MAP: Name which truth-claims are at stake and which were conflated. "
            f"Under 'truth' we often mix: statement-status (is it factually correct?), "
            f"validity-order (who gets to define it?), epistemic procedure (how do we check it?), "
            f"and psychic certainty (does it feel true?). These are not the same. Naming the difference IS the result.\n\n"
            f"PART 2 — BRIDGES & INCOMMENSURABILITY: Where can you translate across methods (different fingers, closer moon)? "
            f"What remains genuinely incommensurable — hold it, don't dissolve it.\n\n"
            f"PART 3 — PROVISIONAL VERDICT: Give a working orientation for this specific question — not a final answer. "
            f"Which claim-type(s) are actually at stake here? What is the most honest working conclusion? "
            f"What remains open? 2-3 sentences, directly usable.\n\n"
            f"PART 4 — DECISION AIDS: For THIS question specifically, which method is the right tool and when? "
            f"Use the pattern: Fact-question → Empirical | Recognition-question → Depth-psychological | "
            f"Power/context-question → Systemic | Concept-question → Philosophical. Make it specific.\n\n"
            f"PART 5 — SUMMARY: 5-6 crisp bullet-point takeaways from THIS analysis — concrete findings, not generic wisdom.\n\n"
            f"PART 6 — PLAIN LANGUAGE ('Einfach gesagt'): Summarize the entire analysis in 3-5 jargon-free sentences "
            f"that anyone can understand. Core question, key finding, what to do with it.\n\n"
            f"PART 7 — HEART-PERSON (herzmensch): What does the heart-person perspective say about this question? "
            f"Speak to feelings, relationships, intuition, meaning. 2-3 warm sentences. Not advice — resonance.\\n\\n"
            f"PART 8 — HEAD-PERSON (kopfmensch): What does the head-person perspective say about this question? "
            f"Speak to logic, facts, strategy, consequences. 2-3 clear, structured sentences.\\n\\n"
            f"PART 9 — MASCULINE ENERGY (maennlich): Not about gender — archetypal masculine qualities: "
            f"action, clarity, structure, focus, decisiveness, forward movement. "
            f"What does this energy say about this question? 2-3 sentences.\\n\\n"
            f"PART 10 — FEMININE ENERGY (weiblich): Not about gender — archetypal feminine qualities: "
            f"care, connection, intuition, receptivity, wholeness, holding complexity. "
            f"What does this energy say about this question? 2-3 sentences.\\n\\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES:\n{perspectives_text}\n\n"
            f"FRICTION:\n{friction_text}\n\n"
            f"Be BRIEF — 1 sentence per item. Respond in English.\n\n{stil_instr}"
        )
        system = (
            "Mapping and verdict agent. Cartography first, then provisional verdict, then decision aids, then summary, then plain-language summary, then heart-person, then head-person, then masculine energy, then feminine energy. "
            "Not synthesis — honest orientation. Short precise sentences. Respond in English."
        )
    else:
        user_content = (
            f"Du bist Kartierungs- und Fazit-Agent. Deine Aufgabe hat zehn Teile:\n\n"
            f"TEIL 1 — ANSPRUCHSKARTE: Benenne, welche Wahrheitsansprüche wirklich vorliegen und welche vermischt wurden. "
            f"Unter 'Wahrheit' vermengen wir oft: Aussagenstatus (stimmt es faktisch?), "
            f"Geltungsordnung (wer darf es definieren?), Erkenntnisverfahren (wie prüfen wir es?), "
            f"und psychische Gewissheit (fühlt es sich wahr an?). Das ist nicht dasselbe. Das Benennen IST das Ergebnis.\n\n"
            f"TEIL 2 — BRÜCKEN & UNVEREINBARKEIT: Wo kann man zwischen Methoden übersetzen (verschiedene Finger, näher am Mond)? "
            f"Was bleibt wirklich unübersetzbar — halte es, löse es nicht auf.\n\n"
            f"TEIL 3 — VORLÄUFIGES FAZIT: Gib eine Arbeitsorientierung für diese spezifische Frage — kein Endurteil. "
            f"Welche(r) Anspruchstyp(en) liegt wirklich vor? Was ist die ehrlichste Arbeitsschlussfolgerung? "
            f"Was bleibt offen? 2-3 Sätze, direkt verwendbar.\n\n"
            f"TEIL 4 — ENTSCHEIDUNGSHILFE: Für DIESE Frage konkret: welche Methode ist das richtige Werkzeug und wann? "
            f"Nutze das Muster: Faktenfrage → Empirie | Anerkennungsfrage → Tiefenpsychologie | "
            f"Macht-/Kontextfrage → Systemik | Begriffsfrage → Philosophie. Spezifisch, nicht generisch.\n\n"
            f"TEIL 5 — KURZFASSUNG: 5-6 knappe Stichpunkte aus DIESER Analyse — konkrete Befunde, keine allgemeine Weisheit.\n\n"
            f"TEIL 6 — EINFACH GESAGT: Fasse die gesamte Analyse in 3-5 klaren, jargonfreien Sätzen zusammen, "
            f"die jede Person verstehen kann. Kernfrage, Kernbefund, was man damit anfangen kann.\n\n"
            f"TEIL 7 — HERZMENSCH: Was sagt die Herzmensch-Perspektive zu dieser Frage? "
            f"Spreche Gefühle, Beziehungen, Intuition und Bedeutung an. 2-3 warme Sätze. Nicht Ratschlag — Resonanz.\\n\\n"
            f"TEIL 8 — KOPFMENSCH: Was sagt die Kopfmensch-Perspektive zu dieser Frage? "
            f"Spreche Logik, Fakten, Strategie und Konsequenzen an. 2-3 klare, strukturierte Sätze.\\n\\n"
            f"TEIL 9 — MÄNNLICHE ENERGIE (maennlich): Nicht Geschlecht — archetypische männliche Qualitäten: "
            f"Handlung, Klarheit, Struktur, Fokus, Entschlossenheit, Vorwärtsbewegung. "
            f"Was sagt diese Energie zu dieser Frage? 2-3 Sätze.\\n\\n"
            f"TEIL 10 — WEIBLICHE ENERGIE (weiblich): Nicht Geschlecht — archetypische weibliche Qualitäten: "
            f"Fürsorge, Verbindung, Intuition, Empfänglichkeit, Ganzheit, Komplexität halten. "
            f"Was sagt diese Energie zu dieser Frage? 2-3 Sätze.\\n\\n"
            f"FRAGE: {question}\\n\\n"
            f"PERSPEKTIVEN:\n{perspectives_text}\n\n"
            f"REIBUNG:\n{friction_text}\n\n"
            f"Sei KURZ — je 1 Satz pro Item.\n\n{stil_instr}"
        )
        system = (
            "Kartierungs- und Fazit-Agent. Erst Kartographie, dann vorläufiges Fazit, dann Entscheidungshilfe, dann Kurzfassung, dann Einfach gesagt, dann Herzmensch, dann Kopfmensch, dann Männliche Energie, dann Weibliche Energie. "
            "Keine Synthese — ehrliche Orientierung. Kurze präzise Sätze."
        )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=system,
        tools=[INTEGRATION_TOOL],
        tool_name="submit_integration",
        messages=[{"role": "user", "content": user_content}]
    )

# ==========================================
# ASYNC WRAPPERS
# ==========================================
async def fetch_perspective(role: str, system_prompt: str, question: str, stil: str = "philosophisch", lang: str = "de") -> Perspective:
    data = await asyncio.to_thread(sync_call_perspective, system_prompt, question, stil, lang)
    data["rolle"] = role
    # Ensure all required fields exist (defensive fallbacks)
    data.setdefault("kernanalyse", data.get("analyse", data.get("core_analysis", "—")))
    data.setdefault("anspruchstyp", data.get("claim_type", "—"))
    data.setdefault("evidenz", data.get("evidence", "—"))
    data.setdefault("blinder_fleck", data.get("blind_spot", "—"))
    # Ensure no field is an empty string — inject fallback explanations
    fb = EMPTY_FALLBACKS_EN if lang == "en" else EMPTY_FALLBACKS_DE
    if not data.get("kernanalyse", "").strip():
        data["kernanalyse"] = "Keine Kernanalyse formulierbar — die Frage liegt außerhalb des methodischen Zuständigkeitsbereichs dieser Perspektive." if lang == "de" else "No core analysis formulable — the question lies outside this perspective's methodological scope."
    if not data.get("anspruchstyp", "").strip():
        data["anspruchstyp"] = "Anspruchstyp nicht eindeutig zuordenbar." if lang == "de" else "Claim type not clearly assignable."
    if not data.get("evidenz", "").strip():
        data["evidenz"] = "Keine Evidenz benennbar — die Frage ist empirisch nicht direkt zugänglich." if lang == "de" else "No evidence nameable — the question is not directly empirically accessible."
    if not data.get("blinder_fleck", "").strip():
        data["blinder_fleck"] = "Blinder Fleck dieser Methode konnte nicht explizit benannt werden." if lang == "de" else "Blind spot of this method could not be explicitly named."
    return Perspective(**data)

async def fetch_friction(perspectives: List[Perspective], question: str, lang: str = "de", stil: str = "philosophisch") -> Friction:
    context = "\n".join([
        f"[{p.rolle}] Anspruchstyp: {p.anspruchstyp[:120]} | Analyse: {p.kernanalyse[:160]} | Blind: {p.blinder_fleck[:80]}"
        for p in perspectives
    ])
    data = await asyncio.to_thread(sync_call_friction, context, question, lang, stil)
    data.setdefault("uebersetzungsfehler", data.get("translation_errors", data.get("scheinkonsens", [])))
    data.setdefault("echte_widersprueche", data.get("genuine_contradictions", data.get("harte_widersprueche", [])))
    data.setdefault("uebersehenes", data.get("overlooked", ""))
    fb = EMPTY_FALLBACKS_EN if lang == "en" else EMPTY_FALLBACKS_DE
    # Ensure lists are never empty
    if not data.get("uebersetzungsfehler"):
        data["uebersetzungsfehler"] = fb["uebersetzungsfehler"]
    if not data.get("echte_widersprueche"):
        data["echte_widersprueche"] = fb["echte_widersprueche"]
    if not str(data.get("uebersehenes", "")).strip():
        data["uebersehenes"] = fb["uebersehenes"]
    return Friction(**data)

async def fetch_integration(perspectives: List[Perspective], friction: Friction, question: str, lang: str = "de", stil: str = "philosophisch") -> Integration:
    perspectives_text = "\n".join([
        f"[{p.rolle}] ({p.anspruchstyp[:80]}): {p.kernanalyse[:150]}"
        for p in perspectives
    ])
    friction_text = (
        f"Übersetzungsfehler: {'; '.join(friction.uebersetzungsfehler[:2])}\n"
        f"Echte Widersprüche: {'; '.join(friction.echte_widersprueche[:2])}\n"
        f"Übersehen: {friction.uebersehenes[:150]}"
    )
    data = await asyncio.to_thread(sync_call_integration, perspectives_text, friction_text, question, lang, stil)
    # Defensive fallbacks for all required fields
    data.setdefault("anspruchskarte", data.get("claim_map", ""))
    data.setdefault("uebersetzbare_bruecken", data.get("translatable_bridges", data.get("fruchtbare_differenzen", [])))
    data.setdefault("echte_unvereinbarkeiten", data.get("genuine_incompatibilities", []))
    data.setdefault("praktische_optionen", data.get("practical_options", []))
    data.setdefault("offene_pruefpfade", data.get("open_paths", []))
    data.setdefault("vorlaeufiges_fazit", data.get("provisional_verdict", ""))
    data.setdefault("entscheidungshilfe", data.get("decision_aids", []))
    data.setdefault("kurzfassung", data.get("summary", []))
    data.setdefault("einfach_gesagt", data.get("plain_summary", data.get("simply_put", "")))
    data.setdefault("herzmensch", data.get("heart_person", ""))
    data.setdefault("kopfmensch", data.get("head_person", ""))
    data.setdefault("maennlich", data.get("masculine", data.get("masculine_energy", "")))
    data.setdefault("weiblich", data.get("feminine", data.get("feminine_energy", "")))

    fb = EMPTY_FALLBACKS_EN if lang == "en" else EMPTY_FALLBACKS_DE
    # Ensure no field is empty — inject fallbacks
    if not str(data.get("anspruchskarte", "")).strip():
        data["anspruchskarte"] = fb["anspruchskarte"]
    if not data.get("uebersetzbare_bruecken"):
        data["uebersetzbare_bruecken"] = fb["uebersetzbare_bruecken"]
    if not data.get("echte_unvereinbarkeiten"):
        data["echte_unvereinbarkeiten"] = fb["echte_unvereinbarkeiten"]
    if not data.get("praktische_optionen"):
        data["praktische_optionen"] = fb["praktische_optionen"]
    if not data.get("offene_pruefpfade"):
        data["offene_pruefpfade"] = fb["offene_pruefpfade"]
    if not str(data.get("vorlaeufiges_fazit", "")).strip():
        data["vorlaeufiges_fazit"] = fb["vorlaeufiges_fazit"]
    if not data.get("entscheidungshilfe"):
        data["entscheidungshilfe"] = fb["entscheidungshilfe"]
    if not data.get("kurzfassung"):
        data["kurzfassung"] = fb["kurzfassung"]
    if not str(data.get("einfach_gesagt", "")).strip():
        data["einfach_gesagt"] = fb["einfach_gesagt"]
    if not str(data.get("herzmensch", "")).strip():
        data["herzmensch"] = "Das Herz hält inne. Was auch immer die Analyse ergeben hat — es gibt etwas in dieser Frage, das mehr ist als Argumente." if lang == "de" else "The heart pauses here. Whatever the analysis found — there is something in this question that is more than arguments."
    if not str(data.get("kopfmensch", "")).strip():
        data["kopfmensch"] = "Der Kopf braucht mehr Informationen für eine belastbare Einschätzung. Die vorliegenden Daten reichen für eine definitive Bewertung nicht aus." if lang == "de" else "The head needs more information for a reliable assessment. The available data is insufficient for a definitive evaluation."
    if not str(data.get("maennlich", "")).strip():
        data["maennlich"] = "Die männliche Energie fragt: Was ist jetzt zu tun? Welcher Schritt bringt Klarheit — unabhängig von Unsicherheit?" if lang == "de" else "The masculine energy asks: What is to be done now? Which step brings clarity — regardless of uncertainty?"
    if not str(data.get("weiblich", "")).strip():
        data["weiblich"] = "Die weibliche Energie fragt: Was darf gehört werden, bevor entschieden wird? Welche Verbindung, welches Gefühl, welche Beziehung spricht hier?" if lang == "de" else "The feminine energy asks: What deserves to be heard before deciding? Which connection, which feeling, which relationship speaks here?"

    return Integration(**data)

# ==========================================
# API ENDPOINTS
# ==========================================
class QueryRequest(BaseModel):
    question: str
    lang: str = "de"   # "de" or "en"
    stil: str = "philosophisch"  # philosophisch | akademisch | alltag | oekonomisch | kindgerecht | therapeutisch

class CustomPerspective(BaseModel):
    name: str        # Name/Rolle der Partei, z.B. "Mein Chef" oder "Teil von mir der Sicherheit will"
    position: str    # Was diese Partei sagt/will/vertritt

class TableRequest(BaseModel):
    question: str
    lang: str = "de"
    stil: str = "philosophisch"
    custom_perspectives: List[CustomPerspective] = []   # 0–4 eigene Perspektiven
    methods: List[str] = []                             # z.B. ["Philosophisch", "Systemisch"] — leere Liste = alle 8

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TEAM TiSCH API", "version": "5.6"}

@app.post("/api/ask", response_model=TableResponse)
async def ask_the_table(req: QueryRequest):
    """Original-Endpunkt: immer alle 8 Methoden-Agenten."""
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short.")

    valid_stile = {"philosophisch", "akademisch", "alltag", "oekonomisch", "kindgerecht", "therapeutisch"}
    stil = req.stil if req.stil in valid_stile else "philosophisch"
    agents = AGENTS_EN if req.lang == "en" else AGENTS_DE

    try:
        tasks = [fetch_perspective(role, prompt, req.question, stil, req.lang) for role, prompt in agents.items()]
        perspectives = list(await asyncio.gather(*tasks))
        friction = await fetch_friction(perspectives, req.question, req.lang, stil)
        integration = await fetch_integration(perspectives, friction, req.question, req.lang, stil)
        return TableResponse(perspectives=perspectives, friction=friction, integration=integration)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")


def build_custom_agent_prompt(cp: CustomPerspective, lang: str) -> str:
    """Baut einen System-Prompt für eine eigene Perspektive (Name + Position)."""
    if lang == "en":
        return (
            f"You are speaking FROM the perspective of: '{cp.name}'.\n"
            f"This perspective's stated position is: '{cp.position}'\n\n"
            f"Your task: Analyze the question STRICTLY from within this perspective's worldview, values, and assumptions. "
            f"Do NOT judge this perspective from outside — inhabit it fully.\n"
            f"Uncover: What deep values, needs, or fears drive this position? What does this perspective assume to be true? "
            f"What would threaten it most? What can it genuinely NOT see from where it stands?\n"
            f"Name the claim-type this perspective is making (factual, value-based, identity-based, existential, strategic).\n"
            f"Be precise. Be fair. No straw-manning. Respond in English."
        )
    else:
        return (
            f"Du sprichst AUS der Perspektive von: '{cp.name}'.\n"
            f"Die geäußerte Position dieser Perspektive ist: '{cp.position}'\n\n"
            f"Deine Aufgabe: Analysiere die Frage STRENG aus der Innenperspektive dieser Weltsicht, ihrer Werte und Annahmen. "
            f"Urteile NICHT von außen — bewohne diese Perspektive vollständig.\n"
            f"Decke auf: Welche tiefen Werte, Bedürfnisse oder Ängste treiben diese Position an? "
            f"Was setzt diese Perspektive als wahr voraus? Was würde sie am meisten bedrohen? "
            f"Was kann sie von ihrem Standpunkt aus genuinely NICHT sehen?\n"
            f"Benenne den Anspruchstyp, den diese Perspektive macht (faktenbasiert, wertbasiert, identitätsbasiert, existenziell, strategisch).\n"
            f"Sei präzise. Sei fair. Kein Strohmann-Argument."
        )


@app.post("/api/ask-table", response_model=TableResponse)
async def ask_the_custom_table(req: TableRequest):
    """Eigener-Tisch-Endpunkt: eigene Perspektiven + optional gewählte Methoden."""
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short.")
    if not req.custom_perspectives and not req.methods:
        raise HTTPException(status_code=400, detail="At least one perspective or method required.")

    valid_stile = {"philosophisch", "akademisch", "alltag", "oekonomisch", "kindgerecht", "therapeutisch"}
    stil = req.stil if req.stil in valid_stile else "philosophisch"

    all_agents_pool = AGENTS_EN if req.lang == "en" else AGENTS_DE

    try:
        tasks = []
        role_sources = []  # "custom" oder "method"

        # 1. Eigene Perspektiven (dynamisch generierte Agenten)
        for cp in req.custom_perspectives[:4]:
            prompt = build_custom_agent_prompt(cp, req.lang)
            # Rolle: Name der Partei, ge-slug-t
            role_label = cp.name.strip() or ("Perspektive" if req.lang == "de" else "Perspective")
            tasks.append(fetch_perspective(role_label, prompt, req.question, stil, req.lang))
            role_sources.append("custom")

        # 2. Gewählte Methoden-Agenten
        if req.methods:
            for method_name in req.methods:
                if method_name in all_agents_pool:
                    tasks.append(fetch_perspective(method_name, all_agents_pool[method_name], req.question, stil, req.lang))
                    role_sources.append("method")
        else:
            # Keine Methoden gewählt = keine Methoden (reine Parteien-Analyse)
            pass

        if not tasks:
            raise HTTPException(status_code=400, detail="No valid perspectives to analyze.")

        perspectives = list(await asyncio.gather(*tasks))

        # Quellinfo in Perspektiven einschreiben (role_source-Tag für Frontend)
        for i, p in enumerate(perspectives):
            # Wir fügen source als Präfix in die rolle, trennbar per Frontend
            src = role_sources[i] if i < len(role_sources) else "method"
            p.rolle = f"[{src}]{p.rolle}"

        friction = await fetch_friction(perspectives, req.question, req.lang, stil)
        integration = await fetch_integration(perspectives, friction, req.question, req.lang, stil)

        return TableResponse(perspectives=perspectives, friction=friction, integration=integration)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}\n\n{tb}")



# ==========================================
# KLÄRUNGSGESPRÄCH — Modell + Endpunkt
# ==========================================
class ClarifyParty(BaseModel):
    name: str = ""
    position: str

class ClarifyRequest(BaseModel):
    question: str
    lang: str = "de"
    stil: str = "philosophisch"
    party_a: ClarifyParty
    party_b: ClarifyParty
    methods: List[str] = []  # Wenn leer → Standard: Pädagogisch + Neurodivergent

@app.post("/api/ask-clarify", response_model=TableResponse)
async def ask_clarify(req: ClarifyRequest):
    """Klärungsgespräch-Endpunkt: Zwei Parteien + optionale Methoden."""
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short.")
    if not req.party_a.position.strip():
        raise HTTPException(status_code=400, detail="Party A position required.")
    if not req.party_b.position.strip():
        raise HTTPException(status_code=400, detail="Party B position required.")

    valid_stile = {"philosophisch", "akademisch", "alltag", "oekonomisch", "kindgerecht", "therapeutisch"}
    stil = req.stil if req.stil in valid_stile else "philosophisch"

    # Parteien als CustomPerspective-Objekte aufbauen
    name_a = req.party_a.name.strip() or ("Partei A" if req.lang == "de" else "Party A")
    name_b = req.party_b.name.strip() or ("Partei B" if req.lang == "de" else "Party B")
    party_a_cp = CustomPerspective(name=name_a, position=req.party_a.position)
    party_b_cp = CustomPerspective(name=name_b, position=req.party_b.position)

    # Methoden: Standard Pädagogisch + Neurodivergent wenn keine gewählt
    methods = req.methods if req.methods else (
        ["Pädagogisch", "Neurodivergent"] if req.lang == "de"
        else ["Pedagogical", "Neurodivergent"]
    )

    # TableRequest zusammenbauen und an ask_the_custom_table weiterleiten
    table_req = TableRequest(
        question=req.question,
        lang=req.lang,
        stil=stil,
        custom_perspectives=[party_a_cp, party_b_cp],
        methods=methods,
    )
    return await ask_the_custom_table(table_req)



# =============================================
# V5.6 — ECOSYSTEM HOOKS (future-proofing)
# Empty routing stubs — NO fake logic
# =============================================

class BlackHoleBoxPayload(BaseModel):
    """Data condenser — receives full session payload for compression."""
    session_id: str
    question: str
    variants: list = []

class KiNtegrityPayload(BaseModel):
    """Translation structure hook — receives bilingual output for structural alignment."""
    text_de: str = ""
    text_en: str = ""
    field_name: str = ""

class MycelPayload(BaseModel):
    """myCEL read/write — accesses Muster-Ordner (pattern directory)."""
    operation: str  # "read" | "write"
    pattern_key: str = ""
    data: dict = {}

class BrainstormzPayload(BaseModel):
    """BRaiNSTORMZ — routing destination for structured idea clouds from myCEL."""
    idea_cloud: list = []
    source_session: str = ""

class PandoraLogicPayload(BaseModel):
    """Pandora_Logic — trigger when idea cloud reaches maximum density."""
    cloud_id: str = ""
    density_score: float = 0.0

@app.post("/api/hooks/black-hole-box", tags=["ecosystem"])
async def hook_black_hole_box(payload: BlackHoleBoxPayload):
    """BLACK-HOLE-BOX: Data condenser hook. Receives full session payload."""
    # HOOKPOINT — integration pending
    return {"status": "hookpoint_ready", "hook": "BLACK-HOLE-BOX", "received": payload.session_id}

@app.post("/api/hooks/ki-ntegrity", tags=["ecosystem"])
async def hook_ki_ntegrity(payload: KiNtegrityPayload):
    """ki-NTEGRiTY: Translation structure hook."""
    # HOOKPOINT — integration pending
    return {"status": "hookpoint_ready", "hook": "ki-NTEGRiTY", "field": payload.field_name}

@app.get("/api/hooks/mycel/patterns", tags=["ecosystem"])
async def hook_mycel_read():
    """myCEL: Read from Muster-Ordner (pattern directory)."""
    # HOOKPOINT — returns empty pattern store until connected
    return {"status": "hookpoint_ready", "hook": "myCEL", "patterns": []}

@app.post("/api/hooks/mycel/patterns", tags=["ecosystem"])
async def hook_mycel_write(payload: MycelPayload):
    """myCEL: Write to Muster-Ordner (pattern directory)."""
    # HOOKPOINT — write operation pending
    return {"status": "hookpoint_ready", "hook": "myCEL", "operation": payload.operation, "key": payload.pattern_key}

@app.post("/api/hooks/brainstormz", tags=["ecosystem"])
async def hook_brainstormz(payload: BrainstormzPayload):
    """BRaiNSTORMZ: Receive structured idea clouds from myCEL."""
    # HOOKPOINT — routing pending
    return {"status": "hookpoint_ready", "hook": "BRaiNSTORMZ", "cloud_size": len(payload.idea_cloud)}

@app.post("/api/hooks/pandora-logic", tags=["ecosystem"])
async def hook_pandora_logic(payload: PandoraLogicPayload):
    """Pandora_Logic: Trigger when idea cloud reaches maximum density."""
    # HOOKPOINT — trigger logic pending
    return {"status": "hookpoint_ready", "hook": "Pandora_Logic", "cloud": payload.cloud_id, "density": payload.density_score}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
