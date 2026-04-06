/**
 * TiSCH NOTiZBUCH — Shared Module
 * Ausklappbares Notizbuch für alle TiSCH Apps
 * localStorage key: tisch_notizbuch_v1
 * 
 * Nutzung: automatisch initialisiert, kein Setup nötig.
 * "Ins Notizbuch" Button: NB.addEntry(text, source)
 */

window.NB = (() => {
  const KEY = 'tisch_notizbuch_v1';
  const MAX = 200;

  // ── Storage ─────────────────────────────────────────────────────────────
  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || '[]'); }
    catch { return []; }
  }
  function save(entries) {
    localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX)));
  }

  // ── Add entry ────────────────────────────────────────────────────────────
  function addEntry(content, source) {
    if (!content || !content.trim()) return;
    const entries = load();
    const entry = {
      id: Date.now(),
      ts: new Date().toISOString(),
      app: document.title.split('—')[0].trim() || 'TiSCH',
      source: source || 'manuell',
      content: content.trim()
    };
    entries.unshift(entry);
    save(entries);
    renderEntries();
    showToast('Ins Notizbuch gespeichert');
    return entry;
  }

  // ── Delete entry ─────────────────────────────────────────────────────────
  function deleteEntry(id) {
    save(load().filter(e => e.id !== id));
    renderEntries();
  }

  // ── Export ───────────────────────────────────────────────────────────────
  function exportTXT() {
    const entries = load();
    const txt = entries.map(e => {
      const d = new Date(e.ts).toLocaleString('de-DE');
      return `[${d}] ${e.app}\n${e.content}\n${'─'.repeat(40)}`;
    }).join('\n\n');
    download(`tisch-notizen-${new Date().toISOString().slice(0,10)}.txt`, txt, 'text/plain');
  }
  function exportJSON() {
    download(`tisch-notizen-${new Date().toISOString().slice(0,10)}.json`,
      JSON.stringify(load(), null, 2), 'application/json');
  }
  function exportCopy() {
    const txt = load().map(e => `${e.app}: ${e.content}`).join('\n\n');
    navigator.clipboard.writeText(txt).then(() => showToast('Kopiert!'));
  }
  function download(name, content, type) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], { type }));
    a.download = name; a.click();
  }

  // ── Toast ────────────────────────────────────────────────────────────────
  function showToast(msg) {
    const t = document.getElementById('nb-toast');
    if (!t) return;
    t.textContent = msg; t.classList.add('nb-toast-show');
    setTimeout(() => t.classList.remove('nb-toast-show'), 2200);
  }

  // ── Render entries ───────────────────────────────────────────────────────
  function renderEntries() {
    const body = document.getElementById('nb-body');
    if (!body) return;
    const entries = load();
    if (!entries.length) {
      body.innerHTML = '<div class="nb-empty">Noch keine Notizen. Ergebnisse oder eigene Gedanken hier speichern.</div>';
      return;
    }
    body.innerHTML = entries.map(e => {
      const d = new Date(e.ts).toLocaleString('de-DE', { day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit' });
      return `<div class="nb-entry" data-id="${e.id}">
        <div class="nb-entry-meta">${e.app} · ${d}</div>
        <div class="nb-entry-text" contenteditable="true" onblur="NB._updateEntry(${e.id}, this.textContent)">${e.content}</div>
        <button class="nb-entry-del" onclick="NB._del(${e.id})" aria-label="Löschen">
          <svg width="12" height="12" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="15" y1="5" x2="5" y2="15"/><line x1="5" y1="5" x2="15" y2="15"/></svg>
        </button>
      </div>`;
    }).join('');
  }

  function _updateEntry(id, text) {
    const entries = load();
    const e = entries.find(x => x.id === id);
    if (e) { e.content = text.trim(); save(entries); }
  }
  function _del(id) { deleteEntry(id); }

  // ── Toggle panel ─────────────────────────────────────────────────────────
  function toggle() {
    const panel = document.getElementById('nb-panel');
    const overlay = document.getElementById('nb-overlay');
    if (!panel) return;
    const open = panel.classList.contains('nb-open');
    panel.classList.toggle('nb-open', !open);
    overlay.classList.toggle('nb-open', !open);
    if (!open) renderEntries();
  }

  // ── "Ins Notizbuch" button for result cards ───────────────────────────────
  function makeButton(content, label) {
    const btn = document.createElement('button');
    btn.className = 'nb-add-btn';
    btn.title = 'Ins Notizbuch';
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h14v14H3z"/><line x1="7" y1="7" x2="13" y2="7"/><line x1="7" y1="10" x2="13" y2="10"/><line x1="7" y1="13" x2="10" y2="13"/></svg> ${label || 'Merken'}`;
    btn.onclick = () => addEntry(content, 'Ergebnis');
    return btn;
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  function init() {
    if (document.getElementById('nb-panel')) return; // already init

    // Detect accent color from CSS var or fallback
    const accent = getComputedStyle(document.documentElement)
      .getPropertyValue('--accent').trim() || '#C9A84C';

    // Inject CSS
    const style = document.createElement('style');
    style.textContent = `
/* ── NOTiZBUCH ───────────────────────────────────────── */
#nb-fab {
  position: fixed; bottom: 28px; right: 28px; z-index: 400;
  width: 48px; height: 48px; border-radius: 50%;
  background: var(--accent, #C9A84C); color: #0D0D0D;
  border: none; cursor: pointer; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.15s, box-shadow 0.15s;
}
#nb-fab:hover { transform: scale(1.08); box-shadow: 0 6px 24px rgba(0,0,0,0.5); }
#nb-fab svg { width: 20px; height: 20px; }

#nb-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.45); z-index: 498; backdrop-filter: blur(2px);
}
#nb-overlay.nb-open { display: block; }

#nb-panel {
  position: fixed; inset-y: 0; right: 0; width: 380px; max-width: 100vw;
  background: var(--surface, #191919); border-left: 1px solid var(--border, #282420);
  z-index: 499; display: flex; flex-direction: column;
  transform: translateX(100%); transition: transform 0.28s cubic-bezier(0.4,0,0.2,1);
  overflow: hidden;
}
#nb-panel.nb-open { transform: translateX(0); }

.nb-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 18px 14px; border-bottom: 1px solid var(--border, #282420);
  flex-shrink: 0;
}
.nb-title {
  font-size: 14px; font-weight: 700; letter-spacing: -0.01em;
  color: var(--text, #F0EDE8); display: flex; align-items: center; gap: 8px;
}
.nb-title svg { color: var(--accent, #C9A84C); }
.nb-actions { display: flex; gap: 6px; }
.nb-action-btn {
  border: 1px solid var(--border, #282420); background: transparent;
  color: var(--muted, #9A9080); border-radius: 7px; padding: 4px 10px;
  font-size: 11px; font-weight: 600; cursor: pointer;
  transition: all 0.15s; font-family: inherit;
}
.nb-action-btn:hover { color: var(--text, #F0EDE8); border-color: var(--border-2, #3A3428); }

.nb-textarea-wrap { padding: 12px 16px; border-bottom: 1px solid var(--border, #282420); flex-shrink: 0; }
.nb-textarea {
  width: 100%; border: 1.5px solid var(--border, #282420);
  border-radius: 10px; padding: 10px 12px; font-family: inherit;
  font-size: 13px; color: var(--text, #F0EDE8); background: var(--surface-2, #141414);
  resize: none; outline: none; min-height: 64px; line-height: 1.55;
  transition: border-color 0.15s;
}
.nb-textarea:focus { border-color: var(--accent, #C9A84C); }
.nb-textarea::placeholder { color: var(--faint, #4A4438); }
.nb-save-btn {
  margin-top: 8px; width: 100%;
  background: var(--accent, #C9A84C); color: #0D0D0D;
  border: none; border-radius: 8px; padding: 8px;
  font-family: inherit; font-size: 13px; font-weight: 700;
  cursor: pointer; transition: opacity 0.15s;
}
.nb-save-btn:hover { opacity: 0.88; }

.nb-body { flex: 1; overflow-y: auto; padding: 12px 16px 20px; }
.nb-empty { font-size: 12px; color: var(--faint, #4A4438); text-align: center; padding: 40px 0; font-style: italic; line-height: 1.6; }
.nb-entry {
  border: 1px solid var(--border, #282420); border-radius: 10px;
  padding: 10px 12px; margin-bottom: 8px; position: relative;
  transition: border-color 0.15s;
}
.nb-entry:hover { border-color: var(--border-2, #3A3428); }
.nb-entry-meta { font-size: 10px; color: var(--faint, #4A4438); margin-bottom: 5px; }
.nb-entry-text {
  font-size: 13px; line-height: 1.6; color: var(--muted, #9A9080);
  outline: none; white-space: pre-wrap;
}
.nb-entry-text:focus { color: var(--text, #F0EDE8); }
.nb-entry-del {
  position: absolute; top: 8px; right: 8px;
  background: none; border: none; cursor: pointer;
  color: var(--faint, #4A4438); opacity: 0; transition: opacity 0.15s;
  display: flex; align-items: center;
}
.nb-entry:hover .nb-entry-del { opacity: 1; }
.nb-entry-del:hover { color: #C05050; }

.nb-close-btn {
  background: none; border: none; cursor: pointer;
  color: var(--muted, #9A9080); padding: 4px;
  display: flex; align-items: center; transition: color 0.15s;
}
.nb-close-btn:hover { color: var(--text, #F0EDE8); }
.nb-close-btn svg { width: 16px; height: 16px; }

.nb-add-btn {
  display: inline-flex; align-items: center; gap: 5px;
  background: none; border: 1px solid var(--border, #282420);
  border-radius: 7px; padding: 4px 10px; font-family: inherit;
  font-size: 11px; font-weight: 600; color: var(--muted, #9A9080);
  cursor: pointer; transition: all 0.15s;
}
.nb-add-btn:hover { border-color: var(--accent, #C9A84C); color: var(--accent, #C9A84C); }

#nb-toast {
  position: fixed; bottom: 86px; right: 28px; z-index: 600;
  background: var(--surface, #191919); border: 1px solid var(--border, #282420);
  border-radius: 8px; padding: 8px 16px; font-size: 12px; font-weight: 600;
  color: var(--text, #F0EDE8); opacity: 0; transform: translateY(8px);
  transition: opacity 0.2s, transform 0.2s; pointer-events: none;
}
#nb-toast.nb-toast-show { opacity: 1; transform: translateY(0); }

@media (max-width: 600px) {
  #nb-panel { width: 100vw; }
  #nb-fab { bottom: 20px; right: 16px; }
}
`;
    document.head.appendChild(style);

    // Inject HTML
    const html = document.createElement('div');
    html.innerHTML = `
<div id="nb-overlay" onclick="NB.toggle()"></div>
<aside id="nb-panel" role="dialog" aria-label="NOTiZBUCH">
  <div class="nb-header">
    <div class="nb-title">
      <svg width="15" height="15" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h14v14H3z"/><line x1="7" y1="7" x2="13" y2="7"/><line x1="7" y1="10" x2="13" y2="10"/><line x1="7" y1="13" x2="10" y2="13"/></svg>
      NOTiZBUCH
    </div>
    <div style="display:flex;align-items:center;gap:6px;">
      <div class="nb-actions">
        <button class="nb-action-btn" onclick="NB.exportCopy()">Kopieren</button>
        <button class="nb-action-btn" onclick="NB.exportTXT()">TXT</button>
        <button class="nb-action-btn" onclick="NB.exportJSON()">JSON</button>
      </div>
      <button class="nb-close-btn" onclick="NB.toggle()" aria-label="Schließen">
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="15" y1="5" x2="5" y2="15"/><line x1="5" y1="5" x2="15" y2="15"/></svg>
      </button>
    </div>
  </div>
  <div class="nb-textarea-wrap">
    <textarea class="nb-textarea" id="nb-input" placeholder="Eigene Notiz eingeben…" rows="3"></textarea>
    <button class="nb-save-btn" onclick="NB._saveInput()">Speichern</button>
  </div>
  <div class="nb-body" id="nb-body"></div>
</aside>
<button id="nb-fab" onclick="NB.toggle()" aria-label="NOTiZBUCH öffnen" title="NOTiZBUCH">
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 3h14v14H3z"/><line x1="7" y1="7" x2="13" y2="7"/><line x1="7" y1="10" x2="13" y2="10"/><line x1="7" y1="13" x2="10" y2="13"/>
  </svg>
</button>
<div id="nb-toast"></div>
`;
    document.body.appendChild(html);

    // Keyboard shortcut: N key (when not in input)
    document.addEventListener('keydown', e => {
      if (e.key === 'n' && !['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) {
        toggle();
      }
    });
  }

  function _saveInput() {
    const input = document.getElementById('nb-input');
    if (!input || !input.value.trim()) return;
    addEntry(input.value, 'manuell');
    input.value = '';
  }

  // Auto-init on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { toggle, addEntry, makeButton, exportTXT, exportJSON, exportCopy, _saveInput, _updateEntry, _del };
})();
