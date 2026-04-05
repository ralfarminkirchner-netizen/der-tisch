"""
kiNTEGRiTY — Shared Core Synthesis Module
Integrity-preserving condensation, synthesis, differentiation, and structured
separation engine for the entire MiNDLAXY app family.

Used by: TiSCH-Familie (TEAM, iNTEGRATiON, DER, EXPERTiSEN, LiTERATUR),
         MOONFiNGERS, BRaiNSTORM SPiRAL, EiN SEiN, and others.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union
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
    # ── MOONFiNGERS ──────────────────────────────────────────
    # Kontemplativ-differenzierend: maximale Differenzsensibilität,
    # Originalsprache heiliger Texte bewahren, Brücken nur bei
    # nachgewiesener struktureller Invarianz.
    "moonfingers_profile": {
        "compression_strength":        0.35,
        "reformulation_conservatism":  0.90,
        "redundancy_reduction":        0.40,
        "conflict_visibility":         0.95,
        "must_keep_sensitivity":       0.95,
        "tone":                        "kontemplativ-differenzierend",
        "special_rules": [
            "Traditionen nie gleichsetzen ohne explizite strukturelle Begründung",
            "Oberflächliche Wortähnlichkeit als solche markieren",
            "Inkommensurabilitäten benennen und bewahren, nicht wegerklären",
            "Brücken nur bei nachgewiesener struktureller Invarianz",
            "Erfahrungsdimensionen phänomenologisch beschreiben, nicht interpretativ einebnen",
            "Metaphern verschiedener Traditionen nicht automatisch gleichsetzen",
            "Bei Vergleichen immer die Vergleichsebene explizit benennen",
        ]
    },
    # ── EiN SEiN ──────────────────────────────────────────────
    # Explorativ-kartografisch: für die Erkundung der Geschichte
    # des Seins; bewahrt historische Kontexte und Verbindungen.
    "einsein_profile": {
        "compression_strength":        0.50,
        "reformulation_conservatism":  0.75,
        "redundancy_reduction":        0.55,
        "conflict_visibility":         0.80,
        "must_keep_sensitivity":       0.90,
        "tone":                        "explorativ-kartografisch",
        "special_rules": [
            "Historische Kontexte und Epochenzuordnungen bewahren",
            "Verbindungen zwischen Traditionen explizit benennen und begründen",
            "Gamification-relevante Entdeckungen hervorheben",
            "Fog-of-War-Status respektieren: noch nicht Entdecktes nicht vorwegnehmen",
        ]
    },
    # ── BRaiNSTORM SPiRAL ─────────────────────────────────────
    # Generativ-spiralförmig: maximale Offenheit für neue Ideen,
    # Reibung als produktive Kraft, Synthese ohne Schließung.
    "brainstorm_spiral_profile": {
        "compression_strength":        0.30,
        "reformulation_conservatism":  0.85,
        "redundancy_reduction":        0.35,
        "conflict_visibility":         0.70,
        "must_keep_sensitivity":       0.95,
        "tone":                        "generativ-spiralförmig",
        "special_rules": [
            "Originalformulierungen maximal bewahren — Ideen sind heilig",
            "Keine vorzeitige Schließung von Widersprüchen",
            "Reibung zwischen Ideen als produktive Kraft sichtbar machen",
            "Synthesepotenzial und Novelty-Score für Ideenpaare ausweisen",
            "Ghost-Ideen (implizite, noch nicht formulierte) benennen",
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
# MOONFiNGERS-SPECIFIC MODELS
# ============================================================
class RelationType(str, Enum):
    """The 12 MOONFiNGERS relation types."""
    STRUCTURAL_PROXIMITY        = "structural_proximity"
    ANALOGY                     = "analogy"
    CONTRAST                    = "contrast"
    TENSION                     = "tension"
    TRANSLATABILITY             = "translatability"
    PARTIAL_TRANSLATABILITY     = "partial_translatability"
    BRIDGE_VIA_THIRD            = "bridge_via_third"
    INCOMMENSURABLE             = "incommensurable"
    SAME_METAPHOR_DIFF_ONTOLOGY = "same_metaphor_different_ontology"
    SIMILAR_EFFECT_DIFF_LOGIC   = "similar_effect_different_logic"
    SAME_QUESTION_DIFF_ANSWER   = "same_question_different_answer"
    SAME_PRAXIS_DIFF_FRAMING    = "same_praxis_different_framing"

class ComparisonLevel(str, Enum):
    """The level at which a comparison operates."""
    FUNCTIONAL   = "funktional"
    ONTOLOGICAL  = "ontologisch"
    LINGUISTIC   = "sprachlich"
    HISTORICAL   = "historisch"
    EXPERIENTIAL = "erfahrungsbezogen"
    METAPHORICAL = "metaphorisch"
    STRUCTURAL   = "strukturell"
    PARTIAL      = "teilweise"
    NONE         = "nicht vergleichbar"

class BridgeStatus(str, Enum):
    CONFIRMED   = "gesichert"
    PLAUSIBLE   = "plausibel"
    EXPLORATIVE = "explorativ"
    CONTESTED   = "strittig"

class BridgeAnalysis(BaseModel):
    """Deep analysis of a bridge between two entities."""
    entity_a: str
    entity_b: str
    relation_type: str           # one of RelationType values
    comparison_level: str        # one of ComparisonLevel values
    confidence: float = 0.5
    status: str = "explorativ"   # one of BridgeStatus values
    justification: str = ""
    translation_risk: str = ""
    limitations: str = ""

class Incommensurability(BaseModel):
    """Documents a genuine incommensurability between entities."""
    entities: List[str]
    reason: str
    why_not_resolvable: str = ""

class MoonfingersResponse(BaseModel):
    """Extended kiNTEGRiTY response for MOONFiNGERS with bridge analysis."""
    synthesis: str
    aber_section: str
    questionable: str
    redundancies_removed: List[str]
    provenance: List[ProvenanceEntry]
    must_keep_honored: bool
    confidence: float
    bridge_analysis: List[BridgeAnalysis] = []
    incommensurabilities: List[Incommensurability] = []
    open_questions: List[str] = []

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

    # MOONFiNGERS extended output schema
    is_moonfingers = req.profile == "moonfingers_profile"
    extra_fields = ""
    extra_instructions = ""
    if is_moonfingers:
        extra_fields = """,
  "bridge_analysis": [
    {{
      "entity_a": "...",
      "entity_b": "...",
      "relation_type": "structural_proximity|analogy|contrast|tension|translatability|partial_translatability|bridge_via_third|incommensurable|same_metaphor_different_ontology|similar_effect_different_logic|same_question_different_answer|same_praxis_different_framing",
      "comparison_level": "funktional|ontologisch|sprachlich|historisch|erfahrungsbezogen|metaphorisch|strukturell|teilweise|nicht vergleichbar",
      "confidence": 0.8,
      "status": "gesichert|plausibel|explorativ|strittig",
      "justification": "...",
      "translation_risk": "...",
      "limitations": "..."
    }}
  ],
  "incommensurabilities": [
    {{"entities": ["...", "..."], "reason": "...", "why_not_resolvable": "..."}}
  ],
  "open_questions": ["..."]"""
        extra_instructions = """

MOONFiNGERS-SPEZIFISCH:
6. Für jedes Paar von Eingabeblöcken: Prüfe die Art der Verbindung
   (strukturelle Nähe, Analogie, Kontrast, Spannung, Übersetzbarkeit, etc.)
7. Identifiziere echte Inkommensurabilitäten — und begründe, warum sie nicht auflösbar sind
8. Benenne offene Fragen, die das System nicht beantworten kann
9. Brücken nur bei nachgewiesener struktureller Invarianz, nicht bei Wortähnlichkeit"""

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
5. Faktisch Fragwürdiges SEPARAT ausweisen, nicht löschen{extra_instructions}

AUSGABE: Antworte NUR mit einem JSON-Objekt, keine Prosa davor oder danach.
JSON-Schema:
{{
  "synthesis": "...",
  "aber_section": "...",
  "questionable": "...",
  "redundancies_removed": ["...", "..."],
  "provenance": [{{"synthesis_passage": "...", "source_ids": ["id1"]}}],
  "must_keep_honored": true,
  "confidence": 0.85{extra_fields}
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
async def kintegrity_synthesize(
    req: KintegrityRequest,
) -> Union[KintegrityResponse, MoonfingersResponse]:
    """
    Run kiNTEGRiTY synthesis. Returns MoonfingersResponse when
    profile == 'moonfingers_profile', else standard KintegrityResponse.
    """
    import asyncio

    system_prompt, user_content = build_kintegrity_prompt(req)

    # MOONFiNGERS needs more tokens for bridge analysis
    max_tokens = 4096 if req.profile == "moonfingers_profile" else 2400

    def _call():
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
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

    base_kwargs = dict(
        synthesis=data.get("synthesis", ""),
        aber_section=data.get("aber_section", ""),
        questionable=data.get("questionable", ""),
        redundancies_removed=data.get("redundancies_removed", []),
        provenance=provenance,
        must_keep_honored=data.get("must_keep_honored", True),
        confidence=float(data.get("confidence", 0.8)),
    )

    # MOONFiNGERS: return extended response with bridge analysis
    if req.profile == "moonfingers_profile":
        bridges = [
            BridgeAnalysis(**b) for b in data.get("bridge_analysis", [])
            if isinstance(b, dict)
        ]
        incomm = [
            Incommensurability(**i) for i in data.get("incommensurabilities", [])
            if isinstance(i, dict)
        ]
        return MoonfingersResponse(
            **base_kwargs,
            bridge_analysis=bridges,
            incommensurabilities=incomm,
            open_questions=data.get("open_questions", []),
        )

    return KintegrityResponse(**base_kwargs)
