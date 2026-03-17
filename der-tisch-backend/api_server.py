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
# AGENT SYSTEM PROMPTS
# ==========================================
AGENTS = {
    "Systemisch": "Du bist der systemische Agent. Methode: Wechselwirkungen, Kontexte, Muster, Zirkularität. Isoliere NIE das Individuum — betrachte das Netz. Nenne Systemebenen (Mikro/Meso/Makro). Halte dich kurz und präzise.",
    "Tiefenpsychologisch": "Du bist der tiefenpsychologische Agent. Methode: Unbewusstes, Schutzstrategien, Bindungsmuster. Frage: Was liegt hinter der Oberfläche? Nutze Psychoanalyse und Bindungstheorie. Halte dich kurz und präzise.",
    "Empirisch-Rational": "Du bist der empirisch-rationale Agent. Methode: Kausalität, kognitive Verzerrungen, Evidenz. Benenne wenn Evidenz fehlt. Nutze: Bestätigungsfehler, Verfügbarkeitsheuristik. Halte dich kurz und präzise.",
    "Philosophisch": "Du bist der philosophische Agent. Methode: Analytische Schärfe, Begriffsklärung, Kategorienfehler aufdecken. Zerlege implizite Annahmen. Halte dich kurz und präzise.",
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
    # Fallback: log stop reason
    stop = getattr(response, 'stop_reason', 'unknown')
    raise RuntimeError(f"No tool_use block (tool={tool_name}, stop_reason={stop}). Check max_tokens.")


def sync_call_perspective(system_prompt: str, question: str) -> dict:
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system_prompt,
        tools=[PERSPECTIVE_TOOL],
        tool_name="submit_perspective",
        messages=[{"role": "user", "content": f"Analysiere diese Frage aus deiner Methode (kurz & präzise):\n\nFRAGE: {question}"}]
    )


def sync_call_friction(context: str, question: str) -> dict:
    user_content = (
        f"Du bist Reibungs-Agent. Finde echte Widersprüche — kein Harmonisieren!\n\n"
        f"FRAGE: {question}\n\n"
        f"PERSPEKTIVEN (komprimiert):\n{context}\n\n"
        f"Sei KURZ — je 1 Satz pro Item."
    )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system="Adversarial Agent. Finde Widersprüche. Keine Synthese. Kurze präzise Sätze.",
        tools=[FRICTION_TOOL],
        tool_name="submit_friction",
        messages=[{"role": "user", "content": user_content}]
    )


def sync_call_integration(perspectives_text: str, friction_text: str, question: str) -> dict:
    user_content = (
        f"Du bist Integrations-Agent. Meta-Synthese ohne Kuschelkonsens.\n\n"
        f"FRAGE: {question}\n\n"
        f"PERSPEKTIVEN:\n{perspectives_text}\n\n"
        f"REIBUNG:\n{friction_text}\n\n"
        f"Respektiere die Reibung. Sei KURZ — je 1 Satz pro Item."
    )
    return _call_api(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="Meta-Integrations-Agent. Synthetisiere ohne zu glätten. Kurze präzise Sätze.",
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

async def fetch_friction(perspectives: List[Perspective], question: str) -> Friction:
    context = "\n".join([
        f"[{p.rolle}]: {p.kernanalyse[:180]} | Blind: {p.blinder_fleck[:80]}"
        for p in perspectives
    ])
    data = await asyncio.to_thread(sync_call_friction, context, question)
    return Friction(**data)

async def fetch_integration(perspectives: List[Perspective], friction: Friction, question: str) -> Integration:
    perspectives_text = "\n".join([f"[{p.rolle}]: {p.kernanalyse[:150]}" for p in perspectives])
    friction_text = (
        f"Widersprüche: {'; '.join(friction.harte_widersprueche[:2])}\n"
        f"Übersehenes: {friction.uebersehenes[:150]}"
    )
    data = await asyncio.to_thread(sync_call_integration, perspectives_text, friction_text, question)
    return Integration(**data)

# ==========================================
# API ENDPOINTS
# ==========================================
class QueryRequest(BaseModel):
    question: str

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Der Tisch API"}

@app.post("/api/ask", response_model=TableResponse)
async def ask_the_table(req: QueryRequest):
    if not req.question or len(req.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Frage zu kurz.")

    # Phase 1: 4 Agenten PARALLEL
    tasks = [fetch_perspective(role, prompt, req.question) for role, prompt in AGENTS.items()]
    perspectives = list(await asyncio.gather(*tasks))

    # Phase 2: Reibungsanalyse
    friction = await fetch_friction(perspectives, req.question)

    # Phase 3: Integration
    integration = await fetch_integration(perspectives, friction, req.question)

    return TableResponse(perspectives=perspectives, friction=friction, integration=integration)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
