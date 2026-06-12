// ============================================================
// API SERVICE — Der Tisch Backend
// ============================================================
export const API_BASE_URL = "https://der-tisch-production.up.railway.app";

export async function askFamilientisch(question, perspectives, tone = 'achtsam', lang = 'de') {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      perspectives: perspectives.map(p => ({ id: p.id, role: p.de })),
      stil: 'therapeutisch',
      register: 'familiensystem',
      tone,
      lang,
      app: 'FAMiLiEN TiSCH',
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Fehler ${response.status}`);
  }

  return response.json();
}

export async function askTheTable(question) {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
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
