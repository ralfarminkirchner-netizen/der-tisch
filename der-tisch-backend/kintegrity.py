"""
kiNTEGRiTY — Shared Core Synthesis Module
Integrity-preserving condensation engine for the TiSCH family.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
import anthropic, os, json

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ============================================================
# APP PROFILES
# ============================================================
PROFILES = {
    "expert_tisch_profile": {
        "compression_strength":        0.55,
        "reformulation_conservatism":  0.72,
        "redundancy_reduction":        0.65,
        "conflict_visibility":         0.85,
        "must_keep_sensitivity":       0.90,
        "tone":                        "wissenschaftlich-sachlich",
        "special_rules": [
            "Fachterminologie erhalten",
            "Widersprüche zwischen Fachbereichen sichtbar machen",
            "Quellenangaben wenn vorhanden bewahren",
        ]
    },
    "integrations_tisch_profile": {
        "compression_strength":        0.60,
        "reformulation_conservatism":  0.65,
        "redundancy_reduction":        0.70,
        "conflict_visibility":         0.75,
        "must_keep_sensitivity":       0.85,
        "tone":                        "therapeutisch-integrierend",
        "special_rules": [
            "Perspektivenvielfalt sichtbar lassen",
            "Gegensätze als produktive Spannung rahmen",
        ]
    },
    "brainstorm_profile": {
        "compression_strength":        0.40,
        "reformulation_conservatism":  0.80,
        "redundancy_reduction":        0.50,
        "conflict_visibility":         0.60,
        "must_keep_sensitivity":       0.95,
        "tone":                        "explorativ-offen",
        "special_rules": [
            "Originalformulierungen maximal bewahren",
            "Keine vorzeitige Schließung von Widersprüchen",
        ]
    },
    "literatentisch_profile": {
        "compression_strength":        0.45,
        "reformulation_conservatism":  0.85,
        "redundancy_reduction":        0.55,
        "conflict_visibility":         0.70,
        "must_keep_sensitivity":       0.92,
        "tone":                        "literarisch-präzise",
        "special_rules": [
            "Sprachrhythmus und Stil bewahren",
            "Literarische Formulierungen nicht glätten",
        ]
    },
    "default_profile": {
        "compression_strength":        0.60,
        "reformulation_conservatism":  0.70,
        "redundancy_reduction":        0.65,
        "conflict_visibility":         0.75,
        "must_keep_sensitivity":       0.85,
        "tone":                        "klar und ausgewogen",
        "special_rules": []
    },
    "familientisch_profile": {
        "compression_strength":        0.42,
        "reformulation_conservatism":  0.90,
        "redundancy_reduction":        0.50,
        "conflict_visibility":         0.88,
        "must_keep_sensitivity":       0.96,
        "tone":                        "empathisch-systemisch",
        "special_rules": [
            "Bewahre emotionale Nuancen und Beziehungsdynamiken",
            "Zeige Muster und Wiederholungen über Generationen auf",
            "Ressourcenorientiert: Staerken der Familie hervorheben",
            "Kein Schuldzuweisen, systemische Perspektive halten"
        ]
    },
    "juristisch_profile": {
        "compression_strength":        0.55,
        "reformulation_conservatism":  0.85,
        "redundancy_reduction":        0.60,
        "conflict_visibility":         0.92,
        "must_keep_sensitivity":       0.96,
        "tone":                        "juristisch-präzise",
        "special_rules": [
            "Behalte Fachbegriffe und Paragrafen-Referenzen bei",
            "Zeige widersprechende Rechtspositionen klar auf",
            "Betone: kein Ersatz für individuelle Rechtsberatung"
        ]
    },
    "medizintisch_profile": {
        "compression_strength":        0.50,
        "reformulation_conservatism":  0.80,
        "redundancy_reduction":        0.60,
        "conflict_visibility":         0.90,
        "must_keep_sensitivity":       0.95,
        "tone":                        "klinisch-verständlich",
        "special_rules": [
            "Behalte Fachbegriffe, aber erkläre sie kurz",
            "Zeige unterschiedliche Behandlungsansätze klar auf",
            "Betone den Haftungshinweis: kein Ersatz für Arztbesuch"
        ]
    },
    "coachingtisch_profile": {
        "compression_strength":        0.45,
        "reformulation_conservatism":  0.88,
        "redundancy_reduction":        0.55,
        "conflict_visibility":         0.80,
        "must_keep_sensitivity":       0.95,
        "tone":                        "empathisch-klar",
        "special_rules": [
            "Bewahre die emotionale Nuance jeder Perspektive",
            "Zeige konstruktive Spannungen zwischen Coaching-Ansätzen auf",
            "Priorisiere handlungsrelevante Erkenntnisse"
        ]
    },
    "trainingstisch_profile": {
        "compression_strength":        0.50,
        "reformulation_conservatism":  0.70,
        "redundancy_reduction":        0.60,
        "conflict_visibility":         0.88,
        "must_keep_sensitivity":       0.92,
        "tone":                        "sportlich-präzise",
        "special_rules": [
            "Priorisiere praktisch umsetzbare Empfehlungen",
            "Behalte sportartspezifische Fachbegriffe bei",
            "Zeige Widersprüche zwischen Perspektiven klar auf"
        ]
    },
}

# ============================================================
# PYDANTIC MODELS
# ============================================================
class InputBlock(BaseModel):
    id: str
    source_field: str = ""
    content: str
    role: str = ""            # e.g. "Philosoph", "Kritiker", "Nutzer"
    is_user_authored: bool = False

class KintegrityRequest(BaseModel):
    inputs: List[InputBlock]
    integrity_field: str = ""     # MUST-KEEP content (max priority)
    profile: str = "default_profile"
    lang: str = "de"
    question: str = ""            # original question / context

class ProvenanceEntry(BaseModel):
    synthesis_passage: str
    source_ids: List[str]

class KintegrityResponse(BaseModel):
    synthesis: str
    aber_section: str             # incommensurable / genuine contradictions
    questionable: str             # factually dubious content
    redundancies_removed: List[str]
    provenance: List[ProvenanceEntry]
    must_keep_honored: bool
    confidence: float

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_kintegrity_prompt(req: KintegrityRequest) -> str:
    profile = PROFILES.get(req.profile, PROFILES["default_profile"])
    lang_instr = "Antworte auf Deutsch." if req.lang == "de" else "Respond in English."

    compression   = profile["compression_strength"]
    conservatism  = profile["reformulation_conservatism"]
    redundancy    = profile["redundancy_reduction"]
    conflict_vis  = profile["conflict_visibility"]
    must_keep_s   = profile["must_keep_sensitivity"]
    tone          = profile["tone"]
    rules         = "\n".join(f"- {r}" for r in profile["special_rules"]) or "- Keine Sonderregeln"

    # Build input blocks text
    input_text = "\n\n".join([
        f"[BLOCK {b.id} | Rolle: {b.role or 'unbekannt'} | Nutzertext: {b.is_user_authored}]\n{b.content}"
        for b in req.inputs
    ])

    must_keep_section = ""
    if req.integrity_field.strip():
        must_keep_section = f"""
=== MUST-KEEP FELD (HÖCHSTE PRIORITÄT) ===
Folgender Text wurde vom Nutzer als unverzichtbar markiert.
Er darf NUR MINIMAL geglättet werden. Inhalt, Formulierungen und Intentionen
müssen im Syntheseresultat maximal erhalten bleiben.
Must-Keep-Sensitivität: {must_keep_s:.0%}

{req.integrity_field}
=== ENDE MUST-KEEP ===
"""

    system_prompt = f"""Du bist kiNTEGRiTY — eine integritätswahrende Synthese-Engine.
Du fasst NICHT einfach zusammen. Du destillierst strukturiert:

PROFIL: {req.profile}
TON: {tone}
{lang_instr}

PARAMETER:
- Verdichtungsstärke: {compression:.0%} (wie stark komprimiert wird)
- Reformulierungskonservatismus: {conservatism:.0%} (wie viel Originalsprache erhalten bleibt)
- Redundanzreduktion: {redundancy:.0%}
- Widerspruchssichtbarkeit: {conflict_vis:.0%}
- Must-Keep-Sensitivität: {must_keep_s:.0%}

PROFILREGELN:
{rules}

KERNREGELN:
1. Was ähnlich KLINGT ist nicht automatisch dasselbe — Unterschiede prüfen
2. Was widersprüchlich ist, ist nicht automatisch falsch — separat ausweisen
3. Inkommensurabel ≠ falsch ≠ wertlos
4. Das Syntheseresultat ist KEIN Endresultat, sondern editierbares Arbeitsmaterial
5. Faktisch Fragwürdiges SEPARAT ausweisen, nicht löschen

AUSGABE: Antworte NUR mit einem JSON-Objekt, keine Prosa davor oder danach.
JSON-Schema:
{{
  "synthesis": "...",
  "aber_section": "...",
  "questionable": "...",
  "redundancies_removed": ["...", "..."],
  "provenance": [{{"synthesis_passage": "...", "source_ids": ["id1"]}}],
  "must_keep_honored": true,
  "confidence": 0.85
}}"""

    user_content = f"""AUSGANGSFRAGE / KONTEXT: {req.question or '(nicht angegeben)'}

{must_keep_section}

EINGEHENDE TEXTBLÖCKE:
{input_text}

Bitte synthesiere, verdichte und differenziere gemäß den Parametern.
Gib NUR das JSON zurück."""

    return system_prompt, user_content


# ============================================================
# MAIN SYNTHESIS FUNCTION
# ============================================================
async def kintegrity_synthesize(req: KintegrityRequest) -> KintegrityResponse:
    import asyncio

    system_prompt, user_content = build_kintegrity_prompt(req)

    def _call():
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    raw = await asyncio.to_thread(_call)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw as synthesis
        data = {
            "synthesis": raw,
            "aber_section": "",
            "questionable": "",
            "redundancies_removed": [],
            "provenance": [],
            "must_keep_honored": bool(req.integrity_field),
            "confidence": 0.5,
        }

    provenance = [
        ProvenanceEntry(**p) if isinstance(p, dict) else ProvenanceEntry(synthesis_passage="", source_ids=[])
        for p in data.get("provenance", [])
    ]

    return KintegrityResponse(
        synthesis=data.get("synthesis", ""),
        aber_section=data.get("aber_section", ""),
        questionable=data.get("questionable", ""),
        redundancies_removed=data.get("redundancies_removed", []),
        provenance=provenance,
        must_keep_honored=data.get("must_keep_honored", True),
        confidence=float(data.get("confidence", 0.8)),
    )
