// TiSCH Wiki Tooltip — Wikipedia API integration
// Usage: addWikiTooltip(element, searchTerm)
// Shows a floating tooltip with Wikipedia extract on hover

window.TiSCHWiki = {
  cache: {},
  
  async fetch(term) {
    if (this.cache[term]) return this.cache[term];
    try {
      const url = `https://de.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(term)}`;
      const r = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (!r.ok) throw new Error();
      const d = await r.json();
      const result = {
        title: d.title,
        extract: d.extract?.slice(0, 200) + '...',
        url: d.content_urls?.desktop?.page,
        thumbnail: d.thumbnail?.source
      };
      this.cache[term] = result;
      return result;
    } catch { return null; }
  },

  init() {
    // Create tooltip DOM element
    if (document.getElementById('tisch-wiki-tooltip')) return;
    const tt = document.createElement('div');
    tt.id = 'tisch-wiki-tooltip';
    tt.innerHTML = `
      <div class="twt-inner">
        <div class="twt-img" id="twt-img"></div>
        <div class="twt-content">
          <div class="twt-title" id="twt-title"></div>
          <div class="twt-text" id="twt-text"></div>
          <a class="twt-link" id="twt-link" target="_blank" rel="noopener">Wikipedia →</a>
        </div>
      </div>`;
    document.body.appendChild(tt);
    
    // CSS
    const style = document.createElement('style');
    style.textContent = `
      #tisch-wiki-tooltip {
        position: fixed; z-index: 9999; pointer-events: none;
        opacity: 0; transition: opacity 0.2s; max-width: 300px;
        background: rgba(15,15,20,0.97); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        overflow: hidden;
      }
      #tisch-wiki-tooltip.visible { opacity: 1; pointer-events: auto; }
      .twt-inner { display: flex; gap: 0; }
      .twt-img { width: 72px; flex-shrink: 0; background-size: cover; background-position: center; }
      .twt-img:empty { display: none; }
      .twt-content { padding: 12px 14px; }
      .twt-title { font-size: 12px; font-weight: 700; color: #fff; margin-bottom: 5px; }
      .twt-text { font-size: 11px; line-height: 1.5; color: rgba(255,255,255,0.65); margin-bottom: 6px; }
      .twt-link { font-size: 10px; color: rgba(255,255,255,0.4); text-decoration: none; }
      .twt-link:hover { color: rgba(255,255,255,0.8); }
    `;
    document.head.appendChild(style);
  },

  attach(el, term) {
    this.init();
    const tt = document.getElementById('tisch-wiki-tooltip');
    let hideTimer;

    el.addEventListener('mouseenter', async (e) => {
      clearTimeout(hideTimer);
      const data = await this.fetch(term);
      if (!data) return;
      document.getElementById('twt-title').textContent = data.title;
      document.getElementById('twt-text').textContent = data.extract;
      const link = document.getElementById('twt-link');
      link.href = data.url || '#';
      const imgEl = document.getElementById('twt-img');
      if (data.thumbnail) {
        imgEl.style.backgroundImage = `url(${data.thumbnail})`;
        imgEl.style.display = 'block';
      } else {
        imgEl.style.display = 'none';
      }
      // Position
      const rect = el.getBoundingClientRect();
      tt.style.left = Math.min(rect.left, window.innerWidth - 320) + 'px';
      tt.style.top = (rect.bottom + 8) + 'px';
      tt.classList.add('visible');
    });

    el.addEventListener('mouseleave', () => {
      hideTimer = setTimeout(() => tt.classList.remove('visible'), 300);
    });
    tt.addEventListener('mouseenter', () => clearTimeout(hideTimer));
    tt.addEventListener('mouseleave', () => tt.classList.remove('visible'));
  }
};
