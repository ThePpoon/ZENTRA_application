/* ═══════════════════════════════════════════════
   ZENTRA — Main Application JS
   Router, WebSocket client, global state
   ═══════════════════════════════════════════════ */

const ZENTRA = {
  /* ── State ──────────────────────────────────── */
  state: {
    pipeline: { running: false, source: null },
    modules:  { ppe: 'ok', zone: 'ok', fall: 'ok' },
    alerts:   { total: 0, warning: 0, emergency: 0 },
    uptime:   0,
    last_emergency: null,
    camera_label: 'Camera #1',
  },

  ws: null,
  _wsRetryTimer: null,
  _statusTimer:  null,
  _currentScreen: null,

  /* ── Router ─────────────────────────────────── */
  async navigate(screenId, params = {}) {
    try {
      const res  = await fetch(`/ui/screens/${screenId}.html`);
      const html = await res.text();
      document.getElementById('app').innerHTML = html;
      ZENTRA._currentScreen = screenId;

      // Run the screen's init function if defined
      const fn = window[`init_${screenId}`];
      if (typeof fn === 'function') fn(params);
    } catch (e) {
      console.error('[ZENTRA] navigate error:', screenId, e);
    }
  },

  /* ── WebSocket ──────────────────────────────── */
  connectWS() {
    if (ZENTRA.ws && ZENTRA.ws.readyState < 2) return;

    const ws = new WebSocket('ws://127.0.0.1:7788/ws/stream');
    ZENTRA.ws = ws;

    ws.onopen  = () => { clearTimeout(ZENTRA._wsRetryTimer); };
    ws.onclose = () => { ZENTRA._wsRetryTimer = setTimeout(() => ZENTRA.connectWS(), 2000); };
    ws.onerror = () => { ws.close(); };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        ZENTRA._handleWsMsg(msg);
      } catch (_) {}
    };
  },

  _handleWsMsg(msg) {
    if (msg.type === 'frame') {
      const el = document.getElementById('video-feed');
      if (el) el.src = 'data:image/jpeg;base64,' + msg.data;
    }
    if (msg.type === 'event') {
      if (msg.event === 'status' || msg.modules) {
        if (msg.modules)  ZENTRA.state.modules  = msg.modules;
        if (msg.alerts)   ZENTRA.state.alerts   = msg.alerts;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
      }
      if (msg.event === 'alert') {
        const lvl = msg.level || 'warning';
        ZENTRA.state.alerts.total++;
        if (lvl === 'warning' || lvl === 'alert') ZENTRA.state.alerts.warning++;
        if (lvl === 'emergency') {
          ZENTRA.state.alerts.emergency++;
          ZENTRA.state.last_emergency = msg;
          ZENTRA._showEmergencyBanner(msg);
        }
        ZENTRA._updateAlertCounters();
        ZENTRA._updateModuleStatus();
      }
    }
  },

  /* ── UI Update Helpers ──────────────────────── */
  _updateModuleStatus() {
    const map = { ppe: 'PPE Module', zone: 'Zone Module', fall: 'Fall Module' };
    const m   = ZENTRA.state.modules;
    for (const [key, _label] of Object.entries(map)) {
      const dotEl   = document.getElementById(`dot-${key}`);
      const labelEl = document.getElementById(`lbl-${key}`);
      if (!dotEl) continue;
      const ok = (m[key] === 'ok');
      dotEl.className = 'status-dot ' + (ok ? 'ok' : 'err');
      if (labelEl) {
        labelEl.textContent  = ok ? 'ปกติ' : 'ไม่ปกติ';
        labelEl.className    = 'module-label ' + (ok ? 'ok' : 'err');
      }
    }
  },

  _updateAlertCounters() {
    const a = ZENTRA.state.alerts;
    const el = (id) => document.getElementById(id);
    if (el('cnt-total'))    el('cnt-total').textContent    = a.total;
    if (el('cnt-warning'))  el('cnt-warning').textContent  = a.warning;
    if (el('cnt-emergency'))el('cnt-emergency').textContent= a.emergency;
  },

  _showEmergencyBanner(msg) {
    const banner = document.getElementById('emergency-banner');
    if (!banner) return;
    banner.classList.remove('hidden');
    const msgEl  = banner.querySelector('.emergency-msg');
    const metaEl = banner.querySelector('.emergency-meta');
    if (msgEl)  msgEl.textContent  = msg.message || 'ตรวจพบเหตุฉุกเฉิน';
    if (metaEl) metaEl.textContent = `${msg.timestamp || ''} · ${msg.camera || ZENTRA.state.camera_label}`;
  },

  /* ── Status Poll ─────────────────────────────── */
  startStatusPoll() {
    ZENTRA._statusTimer = setInterval(async () => {
      try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        ZENTRA.state.modules = data.modules ?? ZENTRA.state.modules;
        ZENTRA.state.alerts  = data.alerts  ?? ZENTRA.state.alerts;
        ZENTRA.state.uptime  = data.uptime  ?? 0;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
      } catch (_) {}
    }, 2000);
  },

  stopStatusPoll() { clearInterval(ZENTRA._statusTimer); },

  /* ── Navigation helpers (called from HTML) ───── */
  goTo(screen) { ZENTRA.navigate(screen); },

  /* ── Bootstrap ──────────────────────────────── */
  async init() {
    ZENTRA.navigate('splash');
  },
};

/* ─── Navbar helper (injected into screens that need it) ─── */
function renderNavbar(activeTab) {
  const tabs = [
    { id: 'dashboard',    label: 'Live Dashboard' },
    { id: 'zone_editor',  label: 'Zone Editor'    },
    { id: 'history',      label: 'History'        },
    { id: 'settings',     label: 'Setting'        },
  ];
  const tabsHtml = tabs.map(t =>
    `<button class="nav-tab${t.id === activeTab ? ' active' : ''}"
       onclick="ZENTRA.navigate('${t.id}')">${t.label}</button>`
  ).join('');

  return `
    <nav class="navbar">
      <div class="navbar-brand">
        <img class="brand-logo" src="/ui/assets/logo.png"
             onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"
             alt="ZENTRA">
        <div style="display:none;width:28px;height:28px;background:linear-gradient(135deg,#7ecfff,#9d4edd);
                    clip-path:polygon(50% 0%,100% 38%,82% 100%,18% 100%,0% 38%);
                    border-radius:2px;"></div>
        <div>
          <span class="brand-text">ZENTRA</span>
          <span class="brand-sub">Safety AI System</span>
        </div>
      </div>
      <div class="navbar-tabs">${tabsHtml}</div>
    </nav>`;
}

/* ─── Init ─────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => { ZENTRA.init(); });
