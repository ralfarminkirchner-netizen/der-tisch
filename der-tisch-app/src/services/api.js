// ============================================================
// API SERVICE — FAMiLiEN TiSCH
// Backend: https://der-tisch-production.up.railway.app
// ============================================================

export const API_BASE_URL = "https://der-tisch-production.up.railway.app";

// Familien-spezifische Agenten je Kategorie
const FAMILY_METHODS = {
  all:          ["Familiensystemisch", "Therapeutisch", "Biografisch", "Pädagogisch"],
  erziehung:    ["Familiensystemisch", "Pädagogisch", "Therapeutisch", "Aus Kinderaugen"],
  beziehung:    ["Familiensystemisch", "Therapeutisch", "Biografisch", "Achtsam"],
  generationen: ["Familiensystemisch", "Biografisch", "Systemisch", "Therapeutisch"],
  krise:        ["Familiensystemisch", "Therapeutisch", "Strategisch", "Risiko"],
};

// Exportiert damit HomeScreen die Lade-Labels kennt
export const LOADING_LABELS = FAMILY_METHODS;

/**
 * Stellt eine Frage an den FAMiLiEN TiSCH.
 * Wählt Agenten automatisch anhand der Kategorie.
 */
export async function askFamilyTable(question, category = "all") {
  const methods = FAMILY_METHODS[category] || FAMILY_METHODS.all;
  const response = await fetch(`${API_BASE_URL}/api/ask-table`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      lang: "de",
      stil: "therapeutisch",
      tone: "achtsam",
      custom_perspectives: [],
      methods,
      reibungsintensitaet: "standard",
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Fehler ${response.status}`);
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  return response.ok;
}
