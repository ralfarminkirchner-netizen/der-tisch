// ============================================================
// API SERVICE — Der Tisch Backend
// ============================================================
// Replace this URL with your Railway deployment URL after deploy
// Example: "https://der-tisch-production.up.railway.app"
export const API_BASE_URL = "https://der-tisch-production.up.railway.app";

/**
 * Send a question to Der Tisch and receive all three phases:
 * perspectives, friction, integration.
 */
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
