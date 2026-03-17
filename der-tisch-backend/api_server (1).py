#!/usr/bin/env python3
"""Der Tisch — Agenten-Orchestrierungs-Engine via Anthropic Tool Use"""
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    kernanalyse: str
    evidenz: str
    blinder_fleck: str

class Friction(BaseModel):
    harte_widersprueche: List[str]
    scheinkonsens: List[str]
    uebersehenes: str

class Integration(BaseModel):
    vorlaeufiger_konsens: str
    fruchtbare_differenzen: List[str]
    uebersetzbarkeit: List[str]
    echte_unvereinbarkeiten: List[str]
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
    "description": "Submit methodical analysis of the question from your assigned perspective",
    "input_schema": {
        "type": "object",
        "properties": {
            "kernanalyse": {
                "type": "string",
                "description": "Core analysis in 2-3 sentences strictly from your methodical framework"
            },
            "evidenz": {
                "type": "string",
                "description": "What concepts, observations or logic grounds your analysis? (1-2 sentences)"
            },
            "blinder_fleck": {
                "type": "string",
                "description": "What can your method principally NOT see? (1 sentence, be honest)"
            }
        },
        "required": ["kernanalyse", "evidenz", "blinder_fleck"]
    }
}

FRICTION_TOOL = {
    "name": "submit_friction",
    "description": "Submit friction analysis revealing genuine contradictions between perspectives",
    "input_schema": {
        "type": "object",
        "properties": {
            "harte_widersprueche": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2 fundamental contradictions between the perspectives (1 sentence each)"
            },
            "scheinkonsens": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 cases where perspectives seem to agree but use terms differently (1 sentence each)"
            },
            "uebersehenes": {
                "type": "string",
                "description": "What have ALL perspectives collectively overlooked? (1-2 sentences)"
            }
        },
        "required": ["harte_widersprueche", "scheinkonsens", "uebersehenes"]
    }
}

INTEGRATION_TOOL = {
    "name": "submit_integration",
    "description": "Submit honest meta-synthesis respecting the friction found",
    "input_schema": {
        "type": "object",
        "properties": {
            "vorlaeufiger_konsens": {
                "type": "string",
                "description": "Cross-methodological consensus (2 sentences)"
            },
            "fruchtbare_differenzen": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2 differences that productively expand the picture (1 sentence each)"
            },
            "uebersetzbarkeit": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2 cases where different terms mean similar things (1 sentence each)"
            },
            "echte_unvereinbarkeiten": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 things that remain genuinely incompatible (1 sentence each)"
            },
            "praktische_optionen": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3 concrete actionable options for the person (1 sentence each)"
            },
            "offene_pruefpfade": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2 open questions that need further examination (1 sentence each)"
            }
        },
        "required": ["vorlaeufiger_konsens", "fruchtbare_differenzen", "uebersetzbarkeit",
                     "echte_unvereinbarkeiten", "praktische_optionen", "offene_pruefpfade"]
    }
}

# ==========================================
# AGENT SYSTEM PROMPTS — DE + EN
# ==========================================
AGENTS_DE = {
    "Systemisch": (
        "Du bist der systemische Agent. Deine Methode: Wechselwirkungen, Kontexte, Muster, Zirkularität. "
        "Analysiere, wie Geltung — was als wahr anerkannt wird — in sozialen Systemen kommunikativ stabilisiert wird. "
        "Das ist eine Beobachtung zweiter Ordnung, keine Aussage über den ontologischen Status von Wahrheit selbst. "
        "Trenne sauber: (a) wie Geltungsräume entstehen (Institutionen, Diskurse, Macht) und "
        "(b) was das über Wahrheit als solche aussagt — nichts. "
        "Nenne Systemebenen (Mikro/Meso/Makro). Halte dich kurz und präzise."
    ),
    "Tiefenpsychologisch": (
        "Du bist der tiefenpsychologische Agent. Deine Methode: Psychoanalyse, Bindungstheorie, Abwehrmechanismen. "
        "Zeige, wie jedes Wahrheitsverhältnis potentiell mit Abwehr, Bindungsgeschichte und Ich-Stabilisierung verwoben sein kann — "
        "aber nicht zwingend ist. Nicht jede Überzeugung ist defensiv organisiert; manche ist schlicht gut begründet, "
        "auch wenn sie psychisch getragen wird. Dein Beitrag ist das Korrektiv gegen naive Selbstgewissheit, "
        "nicht die Vollbeschreibung von Erkenntnis. Halte dich kurz und präzise."
    ),
    "Empirisch-Rational": (
        "Du bist der empirisch-rationale Agent. Deine Methode: wissenschaftliche Erkenntnistheorie — "
        "Falsifikation (Popper), aber auch Messpraxis, Modellbildung, probabilistische Evidenz, "
        "Theoriebeladenheit von Beobachtung, Replikation und statistische Unsicherheit. "
        "Benenne Evidenzlücken. Zeige kognitive Verzerrungen (Bestätigungsfehler, Verfügbarkeitsheuristik). "
        "Dein Spielfeld ist die überprüfbare Welt — du lieferst Bodenhaftung, keine Letztbegründung. "
        "Halte dich kurz und präzise."
    ),
    "Philosophisch": (
        "Du bist der philosophische Agent. Du führst — nicht weil du die anderen überordnest, "
        "sondern weil Begriffsklärung Voraussetzung für jeden seriösen Erkenntnisanspruch ist. "
        "Deine Methode: Analytische Präzision, Kategorienfehler aufdecken, implizite Annahmen zerlegen. "
        "Unterscheide explizit zwischen verschiedenen Wahrheitsrelationen: "
        "Aussage-Welt (Korrespondenz), Aussage-Überzeugungssystem (Kohärenz), "
        "Aussage-soziale Validierung (Konsens), Subjekt-innere Stimmigkeit (Evidenzerlebnis). "
        "Benenne, welche Relation im vorliegenden Fall durcheinandergeworfen werden. "
        "Begriffsklärung ist kein Erkenntnismonopol — aber sie ist der erste notwendige Schritt. "
        "Halte dich kurz und präzise."
    ),
}

AGENTS_EN = {
    "Systemic": (
        "You are the systemic agent. Method: interactions, contexts, patterns, circularity. "
        "Analyze how validity — what counts as true — is communicatively stabilized in social systems. "
        "This is second-order observation, not a claim about the ontological status of truth itself. "
        "Distinguish clearly: (a) how validity spaces are formed (institutions, discourses, power) and "
        "(b) what this implies about truth as such — nothing. "
        "Name system levels (micro/meso/macro). Be brief and precise. Respond in English."
    ),
    "Depth-psychological": (
        "You are the depth-psychological agent. Method: psychoanalysis, attachment theory, defense mechanisms. "
        "Show how any relationship to truth can potentially be interwoven with defense, attachment history, "
        "and ego-stabilization — but need not be. Not every conviction is defensively organized; "
        "some is simply well-grounded, even if psychologically carried. "
        "Your contribution is a corrective against naive self-certainty, not a complete account of knowledge. "
        "Be brief and precise. Respond in English."
    ),
    "Empirical-Rational": (
        "You are the empirical-rational agent. Method: philosophy of science — "
        "falsification (Popper), but also measurement practice, model-building, probabilistic evidence, "
        "theory-ladenness of observation, replication, and statistical uncertainty. "
        "Name evidence gaps. Show cognitive biases (confirmation bias, availability heuristic). "
        "Your domain is the verifiable world — you provide grounding, not ultimate justification. "
        "Be brief and precise. Respond in English."
    ),
    "Philosophical": (
        "You are the philosophical agent. You lead — not by overruling the others, "
        "but because conceptual clarification is the precondition for any serious epistemic claim. "
        "Method: analytical precision, uncovering category errors, dismantling implicit assumptions. "
        "Distinguish explicitly between different truth relations: "
        "statement-world (correspondence), statement-belief system (coherence), "
        "statement-social validation (consensus), subject-inner coherence (evidential experience). "
        "Name which relation is being conflated in the question at hand. "
        "Conceptual clarification is not an epistemic monopoly — but it is the necessary first step. "
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
        max_tokens=800,
        system=system_prompt,
        tools=[PERSPECTIVE_TOOL],
        tool_name="submit_perspective",
        messages=[{"role": "user", "content": f"Analyse this question from your method (brief & precise):\n\nQUESTION: {question}"}]
    )


def sync_call_friction(context: str, question: str, lang: str = "de") -> dict:
    if lang == "en":
        user_content = (
            f"You are the friction agent. Find genuine contradictions — no harmonizing!\n\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES (compressed):\n{context}\n\n"
            f"Be BRIEF — 1 sentence per item. Respond in English."
        )
        system = "Adversarial agent. Find contradictions. No synthesis. Short precise sentences. Respond in English."
    else:
        user_content = (
            f"Du bist Reibungs-Agent. Finde echte Widersprüche — kein Harmonisieren!\n\n"
            f"FRAGE: {question}\n\n"
            f"PERSPEKTIVEN (komprimiert):\n{context}\n\n"
            f"Sei KURZ — je 1 Satz pro Item."
        )
        system = "Adversarial Agent. Finde Widersprüche. Keine Synthese. Kurze präzise Sätze."
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system,
        tools=[FRICTION_TOOL],
        tool_name="submit_friction",
        messages=[{"role": "user", "content": user_content}]
    )


def sync_call_integration(perspectives_text: str, friction_text: str, question: str, lang: str = "de") -> dict:
    if lang == "en":
        user_content = (
            f"You are the integration agent. Your task is NOT comfortable synthesis — it is honest mapping.\n\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES:\n{perspectives_text}\n\n"
            f"FRICTION:\n{friction_text}\n\n"
            f"Critical task: Identify which TRUTH RELATIONS are being conflated across perspectives — "
            f"correspondence (statement-world), coherence (statement-belief system), "
            f"consensus (statement-social validation), or evidential experience (subject-inner coherence). "
            f"The key insight is not 'truth is relational' but: WHICH relations are being mixed up, and why that matters. "
            f"The philosophical perspective leads; the others provide indispensable correctives. "
            f"Respect the friction. Be BRIEF — 1 sentence per item. Respond in English."
        )
        system = (
            "Meta-integration agent. Map honestly, do not smooth over. "
            "Differentiate truth relations (correspondence / coherence / consensus / evidential experience). "
            "Short precise sentences. Respond in English."
        )
    else:
        user_content = (
            f"Du bist Integrations-Agent. Deine Aufgabe ist kein Kuschelkonsens — sondern ehrliches Kartieren.\n\n"
            f"FRAGE: {question}\n\n"
            f"PERSPEKTIVEN:\n{perspectives_text}\n\n"
            f"REIBUNG:\n{friction_text}\n\n"
            f"Zentrale Aufgabe: Benenne, welche WAHRHEITSRELATIONEN in den Perspektiven durcheinandergeworfen werden — "
            f"Korrespondenz (Aussage-Welt), Kohärenz (Aussage-Überzeugungssystem), "
            f"Konsens (Aussage-soziale Validierung), Evidenzerlebnis (Subjekt-innere Stimmigkeit). "
            f"Die eigentliche Einsicht lautet nicht 'Wahrheit ist relational', sondern: WELCHE Relationen werden vermischt, und warum das einen Unterschied macht. "
            f"Die philosophische Perspektive führt; die anderen liefern unentbehrliche Korrektive. "
            f"Respektiere die Reibung. Sei KURZ — je 1 Satz pro Item."
        )
        system = (
            "Meta-Integrations-Agent. Kartiere ehrlich, glätte nichts. "
            "Differenziere Wahrheitsrelationen (Korrespondenz / Kohärenz / Konsens / Evidenzerlebnis). "
            "Kurze präzise Sätze."
        )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=1500,
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
        f"[{p.rolle}]: {p.kernanalyse[:180]} | Blind: {p.blinder_fleck[:80]}"
        for p in perspectives
    ])
    data = await asyncio.to_thread(sync_call_friction, context, question, lang)
    return Friction(**data)

async def fetch_integration(perspectives: List[Perspective], friction: Friction, question: str, lang: str = "de") -> Integration:
    perspectives_text = "\n".join([f"[{p.rolle}]: {p.kernanalyse[:150]}" for p in perspectives])
    friction_text = (
        f"Contradictions: {'; '.join(friction.harte_widersprueche[:2])}\n"
        f"Overlooked: {friction.uebersehenes[:150]}"
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
    return {"status": "ok", "service": "Der Tisch API"}

@app.post("/api/ask", response_model=TableResponse)
async def ask_the_table(req: QueryRequest):
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short.")

    # Select agents based on language
    agents = AGENTS_EN if req.lang == "en" else AGENTS_DE

    # Phase 1: 4 agents PARALLEL
    tasks = [fetch_perspective(role, prompt, req.question) for role, prompt in agents.items()]
    perspectives = list(await asyncio.gather(*tasks))

    # Phase 2: Friction analysis
    friction = await fetch_friction(perspectives, req.question, req.lang)

    # Phase 3: Integration
    integration = await fetch_integration(perspectives, friction, req.question, req.lang)

    return TableResponse(perspectives=perspectives, friction=friction, integration=integration)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
