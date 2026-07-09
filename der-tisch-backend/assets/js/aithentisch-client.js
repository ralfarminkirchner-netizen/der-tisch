// assets/js/aithentisch-client.js
// Frontend-only adapter for the canonical AiTHENTiSCH backend.
// No backend logic. No endpoint invention. Target: /api/ask.
(function () {
  'use strict';

  var DEFAULT_BASE = 'http://localhost:8000';
  var STORAGE_KEY = 'aithentisch_api_base';

  function safeGetLocalStorage(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (_) {
      return null;
    }
  }

  function safeSetLocalStorage(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (_) {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }

  function safeRemoveLocalStorage(key) {
    try {
      window.localStorage.removeItem(key);
    } catch (_) {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }

  function normalizeBaseUrl(baseUrl) {
    return String(baseUrl || '').replace(/\/+$/, '');
  }

  function getApiBase() {
    return normalizeBaseUrl(
      safeGetLocalStorage(STORAGE_KEY) ||
      window.AITHENTISCH_API_BASE ||
      DEFAULT_BASE
    );
  }

  function setApiBase(baseUrl) {
    if (!baseUrl || typeof baseUrl !== 'string') {
      throw new Error('AiTHENTiSCH API base must be a non-empty string');
    }
    safeSetLocalStorage(STORAGE_KEY, normalizeBaseUrl(baseUrl));
  }

  function clearApiBase() {
    safeRemoveLocalStorage(STORAGE_KEY);
  }

  async function ask(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new Error('AiTHENTiSCH.ask requires a valid payload object');
    }

    var apiBase = getApiBase();
    var response = await fetch(apiBase + '/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      var text = await response.text().catch(function () { return ''; });
      throw new Error(
        'AiTHENTiSCH /api/ask failed: ' + response.status + ' ' + response.statusText + (text ? ' — ' + text : '')
      );
    }

    return response.json();
  }

  function normalizeTableResponse(data) {
    return {
      perspectives: Array.isArray(data && data.perspectives) ? data.perspectives : [],
      friction: data && data.friction != null ? data.friction : null,
      integration: data && data.integration != null ? data.integration : '',
      raw: data
    };
  }

  function buildPayload(question, options) {
    options = options || {};

    if (typeof question !== 'string' || !question.trim()) {
      throw new Error('AiTHENTiSCH.buildPayload requires a non-empty question string');
    }

    var payload = {
      question: question,
      table: options.table || 'default',
      mode: options.mode || 'standard',
      perspectives: options.perspectives || null,
      friction_intensity: options.frictionIntensity || null,
      metadata: Object.assign({
        source: 'der-tisch-backend',
        page: window.location.pathname
      }, options.metadata || {})
    };

    return Object.assign(payload, options.extra || {});
  }

  var api = {
    getApiBase: getApiBase,
    setApiBase: setApiBase,
    clearApiBase: clearApiBase,
    ask: ask,
    normalizeTableResponse: normalizeTableResponse,
    buildPayload: buildPayload
  };

  // Canonical global.
  window.AiTHENTiSCH = api;

  // Compatibility alias for older spellings.
  window.AiTHENTISCH = api;

  if (window.console && typeof window.console.log === 'function') {
    window.console.log(
      '%c✅ AiTHENTiSCH Frontend Adapter loaded (der-tisch-backend)',
      'color:#00ff9d; font-weight:bold'
    );
  }
})();
