#!/usr/bin/env python3
"""Der Tisch — Agenten-Orchestrierungs-Engine via Anthropic Tool Use
   Version 3: Inkommensurabilität als Methode. Finger zeigen auf den Mond — nicht der Mond.
"""
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import anthropic

app = FastAPI(title="Der Tisch API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = anthropic.Anthropic()

# Serve index.html at root
@app.get("/")
async def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")

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
    "description": "Map the claim-types, build bridges, hold the incommensurability",
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
            }
        },
        "required": ["anspruchskarte", "uebersetzbare_bruecken", "echte_unvereinbarkeiten",
                     "praktische_optionen", "offene_pruefpfade"]
    }
}

# ==========================================
# AGENT SYSTEM PROMPTS — Version 3
# Prinzip: Jeder Agent ist ein Finger, nicht der Mond.
# Jeder benennt zuerst, welchen Anspruchstyp er bearbeiten kann — und welchen nicht.
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


def sync_call_perspective(system_prompt: str, question: str) -> dict:
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=900,
        system=system_prompt,
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


def sync_call_friction(context: str, question: str, lang: str = "de") -> dict:
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
            f"Be BRIEF — 1 sentence per item. Respond in English."
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
            f"Sei KURZ — je 1 Satz pro Item."
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


def sync_call_integration(perspectives_text: str, friction_text: str, question: str, lang: str = "de") -> dict:
    if lang == "en":
        user_content = (
            f"You are the integration agent. Your task is not synthesis — it is honest mapping.\n\n"
            f"CENTRAL TASK: Build the claim-map. Name which truth-claims are actually at stake — "
            f"and which were conflated under the same word. "
            f"The key insight: under 'truth' we mix statement-status, validity-order, epistemic procedure, and psychic certainty. "
            f"These are not the same thing. Naming the difference IS the result.\n\n"
            f"Then: Where can you build bridges across incommensurable methods? "
            f"Many fingers point to the moon — still not the moon, but closer.\n\n"
            f"And: What remains genuinely incommensurable? "
            f"Hold it — do not dissolve it. That is not failure, that is intellectual honesty.\n\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES:\n{perspectives_text}\n\n"
            f"FRICTION:\n{friction_text}\n\n"
            f"Be BRIEF — 1 sentence per item. Respond in English."
        )
        system = (
            "Mapping agent. Not synthesis — cartography. "
            "Distinguish claim-types. Build bridges where possible. Hold incommensurability where necessary. "
            "Short precise sentences. Respond in English."
        )
    else:
        user_content = (
            f"Du bist Kartierungs-Agent. Deine Aufgabe ist keine Synthese — sondern ehrliche Kartierung.\n\n"
            f"ZENTRALE AUFGABE: Baue die Anspruchskarte. Benenne, welche Wahrheitsansprüche in dieser Frage wirklich vorliegen — "
            f"und welche unter demselben Wort vermischt wurden. "
            f"Die eigentliche Einsicht: Unter 'Wahrheit' vermengen wir Aussagenstatus, Geltungsordnung, "
            f"Erkenntnisverfahren und psychische Gewissheit. Das ist nicht dasselbe. Das Benennen des Unterschieds IST das Ergebnis.\n\n"
            f"Dann: Wo kann man trotz Inkommensurabilität Brücken bauen? "
            f"Viele Finger zeigen auf den Mond — immer noch nicht der Mond, aber näher dran.\n\n"
            f"Und: Was bleibt wirklich unübersetzbar? "
            f"Halte es — löse es nicht auf. Das ist kein Fehler, das ist intellektuelle Redlichkeit.\n\n"
            f"FRAGE: {question}\n\n"
            f"PERSPEKTIVEN:\n{perspectives_text}\n\n"
            f"REIBUNG:\n{friction_text}\n\n"
            f"Sei KURZ — je 1 Satz pro Item."
        )
        system = (
            "Kartierungs-Agent. Keine Synthese — Kartographie. "
            "Unterscheide Anspruchstypen. Baue Brücken wo möglich. Halte Inkommensurabilität wo nötig. "
            "Kurze präzise Sätze."
        )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=1600,
        system=system,
        tools=[INTEGRATION_TOOL],
        tool_name="submit_integration",
        messages=[{"role": "user", "content": user_content}]
    )

# ==========================================
# ASYNC WRAPPERS
# ==========================================
async def fetch_perspective(role: str, system_prompt: str, question: str) -> Perspective:
    data = await asyncio.to_thread(sync_call_perspective, system_prompt, question)
    data["rolle"] = role
    return Perspective(**data)

async def fetch_friction(perspectives: List[Perspective], question: str, lang: str = "de") -> Friction:
    context = "\n".join([
        f"[{p.rolle}] Anspruchstyp: {p.anspruchstyp[:120]} | Analyse: {p.kernanalyse[:160]} | Blind: {p.blinder_fleck[:80]}"
        for p in perspectives
    ])
    data = await asyncio.to_thread(sync_call_friction, context, question, lang)
    return Friction(**data)

async def fetch_integration(perspectives: List[Perspective], friction: Friction, question: str, lang: str = "de") -> Integration:
    perspectives_text = "\n".join([
        f"[{p.rolle}] ({p.anspruchstyp[:80]}): {p.kernanalyse[:150]}"
        for p in perspectives
    ])
    friction_text = (
        f"Übersetzungsfehler: {'; '.join(friction.uebersetzungsfehler[:2])}\n"
        f"Echte Widersprüche: {'; '.join(friction.echte_widersprueche[:2])}\n"
        f"Übersehen: {friction.uebersehenes[:150]}"
    )
    data = await asyncio.to_thread(sync_call_integration, perspectives_text, friction_text, question, lang)
    return Integration(**data)

# ==========================================
# API ENDPOINTS
# ==========================================
class QueryRequest(BaseModel):
    question: str
    lang: str = "de"  # "de" or "en"

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Der Tisch API", "version": "3.0"}

@app.post("/api/ask", response_model=TableResponse)
async def ask_the_table(req: QueryRequest):
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short.")

    agents = AGENTS_EN if req.lang == "en" else AGENTS_DE

    # Phase 1: 4 Agenten PARALLEL — jeder benennt seinen Anspruchstyp
    tasks = [fetch_perspective(role, prompt, req.question) for role, prompt in agents.items()]
    perspectives = list(await asyncio.gather(*tasks))

    # Phase 2: Reibung — Übersetzungsfehler vs. echte Widersprüche
    friction = await fetch_friction(perspectives, req.question, req.lang)

    # Phase 3: Kartierung — nicht Synthese
    integration = await fetch_integration(perspectives, friction, req.question, req.lang)

    return TableResponse(perspectives=perspectives, friction=friction, integration=integration)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
