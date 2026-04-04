// =============================================
// DER TiSCH — API Service
// =============================================
const API_BASE = "https://der-tisch-production.up.railway.app";

/**
 * Ask the table (simple mode — KI picks 4–6 agents).
 */
export async function askTheTable(question, options = {}) {
  const { lang = "de", stil = "alltag", tone = "achtsam", register = "" } = options;
  const res = await fetch(`${API_BASE}/api/ask-simple`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, lang, stil, tone, register }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Ask with custom expert perspectives (EXPERTiSEN TiSCH mode).
 * @param {string} question
 * @param {Array<{name:string, position:string}>} experts
 */
export async function askExperts(question, experts, options = {}) {
  const { lang = "de", stil = "akademisch", tone = "achtsam" } = options;
  const res = await fetch(`${API_BASE}/api/ask-table`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      lang,
      stil,
      tone,
      register: "",
      custom_perspectives: experts,
      methods: [],
      reibungsintensitaet: "standard",
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Preview which experts KI would pick for a question.
 * Uses ask-simple but only returns the perspective roles (fast preview).
 * We simulate this with a lightweight call and extract roles client-side.
 */
export async function previewExperts(question) {
  // We ask with a special prompt hint to get expert names quickly
  const res = await fetch(`${API_BASE}/api/ask-simple`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      lang: "de",
      stil: "alltag",
      tone: "achtsam",
      register: "",
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  // Extract just the role names for preview
  return (data.perspectives || []).map((p) => ({
    role: (p.rolle || p.role || "Experte").replace(/^\[(custom|method)\]/, "").trim(),
    snippet: (p.kernanalyse || p.text || "").slice(0, 80),
  }));
}

/**
 * kiNTEGRiTY synthesis.
 */
export async function kintegritySynthesize(inputs, integrityField = "", question = "", lang = "de") {
  const res = await fetch(`${API_BASE}/api/kintegrity/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      inputs,
      integrity_field: integrityField,
      profile: "expert_tisch_profile",
      lang,
      question,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
