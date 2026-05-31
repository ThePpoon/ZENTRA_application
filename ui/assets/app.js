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
    camera:   'disconnected',   // connected | reconnecting | disconnected
    recentAlarms: [],           // [{level,message,time,camera}] newest first
  },

  ws: null,
  _wsRetryTimer: null,
  _statusTimer:  null,
  _currentScreen: null,

  /* ── Router ─────────────────────────────────── */
  async navigate(screenId, params = {}) {
    try {
      const res       = await fetch(`/ui/screens/${screenId}.html`);
      const html      = await res.text();
      const container = document.getElementById('app');
      container.innerHTML = html;
      ZENTRA._currentScreen = screenId;

      // innerHTML does NOT auto-execute <script> tags — re-create them.
      // Re-creating a <script> element makes the browser execute it in
      // GLOBAL scope (eval() would only define functions locally, so
      // window['init_<screen>'] would never be found).
      const scripts = container.querySelectorAll('script');
      const externalLoads = [];

      for (const oldScript of scripts) {
        if (oldScript.src) {
          // External CDN script — load once, append to head, await onload
          if (!document.querySelector(`script[data-cdn="${oldScript.src}"]`)) {
            externalLoads.push(new Promise(resolve => {
              const el = document.createElement('script');
              el.src   = oldScript.src;
              el.setAttribute('data-cdn', oldScript.src);
              el.onload  = resolve;
              el.onerror = resolve;
              document.head.appendChild(el);
            }));
          }
        } else if (oldScript.textContent.trim()) {
          // Inline script — re-create so it runs in global scope
          const el = document.createElement('script');
          el.textContent = oldScript.textContent;
          document.body.appendChild(el);
          document.body.removeChild(el);
        }
      }

      // Wait for external scripts (e.g. Chart.js) before calling init
      if (externalLoads.length) await Promise.all(externalLoads);

      // Call the screen's init function (now globally defined)
      const fn = window[`init_${screenId}`];
      if (typeof fn === 'function') fn(params);

      // Keep the header clock/status pill running on screens that have a navbar
      ZENTRA.startHeaderClock();
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
      ZENTRA._lastFrame = 'data:image/jpeg;base64,' + msg.data;
      const el = document.getElementById('video-feed');
      if (el) el.src = ZENTRA._lastFrame;
    }
    if (msg.type === 'event') {
      if (msg.event === 'status' || msg.modules) {
        if (msg.modules)  ZENTRA.state.modules  = msg.modules;
        if (msg.alerts)   ZENTRA.state.alerts   = msg.alerts;
        if (msg.camera)   ZENTRA.state.camera   = msg.camera;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
        ZENTRA._updateCameraState();
      }
      if (msg.event === 'alert') {
        const lvl = msg.level || 'warning';
        // Prefer authoritative counts from server; fall back to local increment
        if (msg.alerts) {
          ZENTRA.state.alerts = msg.alerts;
        } else {
          ZENTRA.state.alerts.total++;
          if (lvl === 'warning' || lvl === 'alert') ZENTRA.state.alerts.warning++;
          if (lvl === 'emergency') ZENTRA.state.alerts.emergency++;
        }
        // Push to the recent-alarms list (newest first, cap 30)
        ZENTRA.state.recentAlarms.unshift({
          level: lvl, message: msg.message || '', time: msg.timestamp || '', camera: msg.camera || '',
        });
        ZENTRA.state.recentAlarms = ZENTRA.state.recentAlarms.slice(0, 30);
        if (lvl === 'emergency') {
          ZENTRA.state.last_emergency = msg;
          ZENTRA._showEmergencyBanner(msg);
        }
        ZENTRA._updateAlertCounters();
        ZENTRA._updateKPIs();
        ZENTRA._renderAlarms();
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

  _fmtUptime(secs) {
    secs = secs || 0;
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return h > 0 ? `${h}:${String(m).padStart(2,'0')}` : `${m} น.`;
  },

  _updateKPIs() {
    const a = ZENTRA.state.alerts || {};
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('kpi-total',     a.total     || 0);
    set('kpi-warning',   a.warning   || 0);
    set('kpi-emergency', a.emergency || 0);
    set('kpi-uptime',    ZENTRA._fmtUptime(ZENTRA.state.uptime));
    const m = ZENTRA.state.modules || {};
    const ok = ['ppe','zone','fall'].filter(k => m[k] === 'ok').length;
    set('kpi-modules', `${ok}/3`);
    // Color tiles only when there is something to show (control-room style)
    const wt = document.getElementById('kpi-tile-warning');
    const et = document.getElementById('kpi-tile-emergency');
    if (wt) wt.classList.toggle('warn',  (a.warning   || 0) > 0);
    if (et) et.classList.toggle('alarm', (a.emergency || 0) > 0);
  },

  _renderAlarms() {
    const list = document.getElementById('alarm-list');
    if (!list) return;
    const items = ZENTRA.state.recentAlarms || [];
    if (!items.length) {
      list.innerHTML = '<div class="alarm-empty" id="alarm-empty">ยังไม่มีการแจ้งเตือน</div>';
      return;
    }
    list.innerHTML = items.map(it => `
      <div class="alarm-item ${it.level || 'warning'}">
        <div class="a-msg">${(it.message || '').replace(/</g,'&lt;')}</div>
        <div class="a-meta">${it.time || ''}${it.camera ? ' · ' + it.camera : ''}</div>
      </div>`).join('');
  },

  _updateCameraState() {
    // Show a connecting/reconnecting overlay over the video feed.
    const overlay = document.getElementById('video-overlay');
    if (!overlay) return;
    const state = ZENTRA.state.camera;
    if (state === 'connected') {
      overlay.classList.add('hidden');
    } else {
      overlay.classList.remove('hidden');
      const txt = overlay.querySelector('.video-overlay-text');
      if (txt) {
        txt.textContent = (state === 'reconnecting')
          ? 'สัญญาณกล้องหลุด — กำลังเชื่อมต่อใหม่...'
          : 'กำลังเชื่อมต่อกล้อง...';
      }
    }
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
    // Clear any existing poll first so repeated dashboard visits don't
    // stack multiple intervals (would multiply /api/status traffic).
    if (ZENTRA._statusTimer) clearInterval(ZENTRA._statusTimer);
    ZENTRA._statusTimer = setInterval(async () => {
      try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        ZENTRA.state.modules = data.modules ?? ZENTRA.state.modules;
        ZENTRA.state.alerts  = data.alerts  ?? ZENTRA.state.alerts;
        ZENTRA.state.uptime  = data.uptime  ?? 0;
        if (data.camera) ZENTRA.state.camera = data.camera;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
        ZENTRA._updateKPIs();
        ZENTRA._updateCameraState();
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
        <span class="brand-mark">Z</span>
        <div>
          <span class="brand-text">ZENTRA</span>
          <span class="brand-sub">Safety AI System</span>
        </div>
      </div>
      <div class="navbar-tabs">${tabsHtml}</div>
      <div class="navbar-status">
        <span class="nav-clock" id="nav-clock">--:--:--</span>
        <span class="sys-pill ok" id="sys-pill"><span class="sys-dot"></span><span id="sys-pill-text">ระบบปกติ</span></span>
      </div>
    </nav>`;
}

/* ─── Header clock + system-status pill ───────────── */
ZENTRA._headerTimer = null;
ZENTRA.startHeaderClock = function () {
  if (ZENTRA._headerTimer) return;
  const tick = () => {
    const el = document.getElementById('nav-clock');
    if (el) {
      const d = new Date();
      el.textContent = d.toLocaleTimeString('th-TH', { hour12: false });
    }
    ZENTRA._updateSysPill();
  };
  tick();
  ZENTRA._headerTimer = setInterval(tick, 1000);
};
ZENTRA._updateSysPill = function () {
  const pill = document.getElementById('sys-pill');
  const txt  = document.getElementById('sys-pill-text');
  if (!pill || !txt) return;
  const a = ZENTRA.state.alerts || {};
  const cam = ZENTRA.state.camera;
  let cls = 'ok', label = 'ระบบปกติ';
  if (cam === 'reconnecting' || cam === 'disconnected') { cls = 'warn'; label = 'กล้องไม่พร้อม'; }
  if ((a.emergency || 0) > 0)                            { cls = 'alarm'; label = 'เหตุฉุกเฉิน'; }
  else if ((a.warning || 0) > 0)                          { cls = 'warn';  label = 'มีการแจ้งเตือน'; }
  pill.className = 'sys-pill ' + cls;
  txt.textContent = label;
};

/* ─── Init ─────────────────────────────────────── */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { ZENTRA.init(); });
} else {
  ZENTRA.init();
}
