/* ═══════════════════════════════════════════════
   CampaignIQ — Dashboard JavaScript
   Handles: navigation, data fetching, Chart.js,
            filtering, pagination, modal, predict
   ═══════════════════════════════════════════════ */

const API = 'http://127.0.0.1:5000/api';

// ─────────────────────────────────────────────
// Chart.js global defaults (dark theme)
// ─────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e1e42';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

const PALETTE = ['#7c3aed','#10b981','#f59e0b','#ef4444','#3b82f6','#ec4899','#14b8a6','#a855f7'];

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let currentPage = 1;
let totalCustomers = 0;
let sortCol = 'response_prob';
let sortDir = 'desc';
let charts = {};
let modalSpendChart = null;
let modalPredChart  = null;
let segmentList = [];

// ─────────────────────────────────────────────
// Utils
// ─────────────────────────────────────────────
const fmt_pct  = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
const fmt_curr = v => v == null ? '—' : '$' + Number(v).toLocaleString('en', {maximumFractionDigits:0});
const fmt_int  = v => v == null ? '—' : Number(v).toLocaleString();

function tierBadge(tier) {
  return `<span class="badge tier-${tier}">${tier}</span>`;
}
function churnBadge(risk) {
  return `<span class="badge churn-${risk}">${risk}</span>`;
}

async function apiFetch(endpoint) {
  try {
    const r = await fetch(API + endpoint);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch(e) {
    console.error('API error:', endpoint, e);
    return null;
  }
}

// ─────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────
const PAGE_META = {
  overview: { title: 'Overview',          subtitle: 'ML-powered customer intelligence' },
  explorer: { title: 'Customer Explorer', subtitle: 'Browse, filter, and inspect scored customers' },
  predict:  { title: 'Predict New',       subtitle: 'Get instant ML predictions for a new customer' },
  insights: { title: 'Model Insights',    subtitle: 'Feature importances and model performance' },
};

function navigate(page) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(`nav-${page}`).classList.add('active');
  document.getElementById(`page-${page}`).classList.add('active');
  document.getElementById('page-title').textContent    = PAGE_META[page].title;
  document.getElementById('page-subtitle').textContent = PAGE_META[page].subtitle;
  document.getElementById('global-search-box').style.display = page === 'explorer' ? 'flex' : 'none';

  if (page === 'insights' && !charts['feat-response']) loadInsights();
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => navigate(btn.dataset.page));
});

// ─────────────────────────────────────────────
// Status check
// ─────────────────────────────────────────────
async function checkStatus() {
  const data = await apiFetch('/overview');
  const dot  = document.getElementById('status-dot');
  const txt  = document.getElementById('status-text');
  if (data) {
    dot.classList.add('online');
    txt.textContent = 'Models online';
  } else {
    txt.textContent = 'Server offline';
  }
  return data;
}

// ─────────────────────────────────────────────
// Overview — KPI cards
// ─────────────────────────────────────────────
async function loadOverview() {
  const data = await apiFetch('/overview');
  if (!data) return;

  animateCount('kpi-total',     data.total_customers,      fmt_int);
  animateCount('kpi-responders',data.predicted_responders, fmt_int);
  animateCount('kpi-ltv',       data.avg_ltv,              fmt_curr);
  animateCount('kpi-churn',     data.high_churn_risk,      fmt_int);
  document.getElementById('kpi-response-rate').textContent = `${data.response_rate_pct}% response rate`;
  document.getElementById('kpi-churn-pct').textContent     = `${data.churn_risk_pct}% of customers`;
}

function animateCount(id, target, formatter) {
  const el = document.getElementById(id);
  if (!el || target == null) return;
  const start = 0;
  const dur   = 1200;
  const begin = performance.now();
  function step(now) {
    const t = Math.min((now - begin) / dur, 1);
    const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
    const val = start + (target - start) * ease;
    el.textContent = formatter(val);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = formatter(target);
  }
  requestAnimationFrame(step);
}

// ─────────────────────────────────────────────
// Overview — Charts
// ─────────────────────────────────────────────
function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

async function loadResponseDist() {
  const data = await apiFetch('/response-dist');
  if (!data || !data.length) return;
  destroyChart('response-dist');
  const labels = data.map(d => d.range);
  const values = data.map(d => d.count);
  const colors = values.map((_, i) => {
    const pct = i / values.length;
    if (pct < 0.3) return 'rgba(100,116,139,0.7)';
    if (pct < 0.6) return 'rgba(245,158,11,0.75)';
    return 'rgba(16,185,129,0.8)';
  });
  const ctx = document.getElementById('chart-response-dist').getContext('2d');
  charts['response-dist'] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, borderSkipped: false }] },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

async function loadSegmentChart() {
  segmentList = await apiFetch('/segments');
  if (!segmentList || !segmentList.length) return;

  destroyChart('segments');
  const ctx = document.getElementById('chart-segments').getContext('2d');
  charts['segments'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: segmentList.map(s => s.name),
      datasets: [{
        data: segmentList.map(s => s.count),
        backgroundColor: PALETTE.slice(0, segmentList.length),
        borderWidth: 2, borderColor: '#111127',
        hoverBorderColor: '#fff',
      }]
    },
    options: {
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 12, font: { size: 11 } } }
      }
    }
  });

  // Populate segment filter in explorer
  const sel = document.getElementById('filter-segment');
  segmentList.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.name; opt.textContent = s.name;
    sel.appendChild(opt);
  });
}

async function loadProductChart() {
  const data = await apiFetch('/top-products');
  if (!data || !data.length) return;
  destroyChart('products');
  const ctx = document.getElementById('chart-products').getContext('2d');
  charts['products'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.product),
      datasets: [{
        data: data.map(d => d.count),
        backgroundColor: PALETTE,
        borderRadius: 6, borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { grid: { display: false } }
      }
    }
  });
}

async function loadSegmentMatrix() {
  if (!segmentList.length) segmentList = await apiFetch('/segments') || [];
  if (!segmentList.length) return;
  destroyChart('segment-matrix');
  const ctx = document.getElementById('chart-segment-matrix').getContext('2d');
  charts['segment-matrix'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: segmentList.map(s => s.name),
      datasets: [
        {
          label: 'Avg Response Prob',
          data: segmentList.map(s => s.avg_response),
          backgroundColor: 'rgba(16,185,129,0.75)',
          borderRadius: 4, borderSkipped: false, yAxisID: 'y',
        },
        {
          label: 'Avg Churn Prob',
          data: segmentList.map(s => s.avg_churn),
          backgroundColor: 'rgba(239,68,68,0.7)',
          borderRadius: 4, borderSkipped: false, yAxisID: 'y',
        },
        {
          label: 'Avg Pred LTV ($)',
          data: segmentList.map(s => s.avg_ltv),
          backgroundColor: 'rgba(245,158,11,0.75)',
          borderRadius: 4, borderSkipped: false, yAxisID: 'y2',
          type: 'bar',
        }
      ]
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { padding: 12, font: { size: 11 } } } },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 20, font: { size: 10 } } },
        y:  { position: 'left',  grid: { color: 'rgba(255,255,255,0.05)' }, max: 1 },
        y2: { position: 'right', grid: { display: false } }
      }
    }
  });
}

// ─────────────────────────────────────────────
// Explorer — Table
// ─────────────────────────────────────────────
function getFilters() {
  return {
    search:        document.getElementById('global-search').value.trim(),
    segment:       document.getElementById('filter-segment').value,
    product:       document.getElementById('filter-product').value,
    response_tier: document.getElementById('filter-response').value,
    churn_risk:    document.getElementById('filter-churn').value,
    channel:       document.getElementById('filter-channel').value,
  };
}

async function loadTable(page = 1) {
  currentPage = page;
  const f = getFilters();
  const params = new URLSearchParams({
    page, per_page: 50, sort_by: sortCol, sort_dir: sortDir,
    ...Object.fromEntries(Object.entries(f).filter(([,v]) => v))
  });

  const tbody = document.getElementById('table-body');
  tbody.innerHTML = `<tr><td colspan="11" class="table-loading">Loading...</td></tr>`;

  const data = await apiFetch(`/customers?${params}`);
  if (!data) { tbody.innerHTML = `<tr><td colspan="11" class="table-loading">Error loading data</td></tr>`; return; }

  totalCustomers = data.total;
  document.getElementById('table-count').textContent =
    `${totalCustomers.toLocaleString()} customer${totalCustomers !== 1 ? 's' : ''} found`;

  tbody.innerHTML = '';
  if (!data.customers.length) {
    tbody.innerHTML = `<tr><td colspan="11" class="table-loading">No customers match the filters</td></tr>`;
  } else {
    data.customers.forEach(c => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${c.ID}</strong></td>
        <td>${c.age ?? '—'}</td>
        <td>${fmt_curr(c.income)}</td>
        <td><strong style="color:${probColor(c.response_prob)}">${fmt_pct(c.response_prob)}</strong></td>
        <td>${tierBadge(c.response_tier ?? '—')}</td>
        <td>${churnBadge(c.churn_risk ?? '—')}</td>
        <td>${fmt_curr(c.predicted_ltv)}</td>
        <td><span class="badge badge-product">${c.top_product ?? '—'}</span></td>
        <td>${c.recommended_channel ?? '—'}</td>
        <td><span class="badge badge-segment" style="font-size:10px">${c.segment_name ?? '—'}</span></td>
        <td><button class="btn-view" data-id="${c.ID}">View</button></td>
      `;
      tbody.appendChild(tr);
    });
  }

  renderPagination(totalCustomers, 50, currentPage);
  attachViewButtons();
}

function probColor(p) {
  if (p == null) return '#94a3b8';
  if (p >= 0.6) return '#10b981';
  if (p >= 0.3) return '#f59e0b';
  return '#64748b';
}

function renderPagination(total, perPage, current) {
  const pages = Math.ceil(total / perPage);
  const el = document.getElementById('pagination');
  el.innerHTML = '';
  const shown = [];
  if (pages <= 7) for (let i=1; i<=pages; i++) shown.push(i);
  else {
    shown.push(1);
    if (current > 3) shown.push('...');
    for (let i=Math.max(2,current-1); i<=Math.min(pages-1,current+1); i++) shown.push(i);
    if (current < pages - 2) shown.push('...');
    shown.push(pages);
  }
  shown.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'page-btn' + (p === current ? ' active' : '');
    btn.textContent = p;
    if (p !== '...') btn.addEventListener('click', () => loadTable(p));
    else { btn.style.cursor = 'default'; btn.style.pointerEvents = 'none'; }
    el.appendChild(btn);
  });
}

function attachViewButtons() {
  document.querySelectorAll('.btn-view[data-id]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      openModal(parseInt(btn.dataset.id));
    });
  });
  // Row click
  document.querySelectorAll('#table-body tr').forEach(tr => {
    tr.addEventListener('click', () => {
      const btn = tr.querySelector('.btn-view');
      if (btn) openModal(parseInt(btn.dataset.id));
    });
  });
}

// Sort headers
document.querySelectorAll('th.sortable').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) sortDir = sortDir === 'desc' ? 'asc' : 'desc';
    else { sortCol = col; sortDir = 'desc'; }
    loadTable(1);
  });
});

// Filter change handlers
['filter-segment','filter-product','filter-response','filter-churn','filter-channel'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => loadTable(1));
});
document.getElementById('global-search').addEventListener('input', debounce(() => loadTable(1), 350));
document.getElementById('btn-clear-filters').addEventListener('click', () => {
  ['filter-segment','filter-product','filter-response','filter-churn','filter-channel']
    .forEach(id => document.getElementById(id).value = '');
  document.getElementById('global-search').value = '';
  loadTable(1);
});

function debounce(fn, ms) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ─────────────────────────────────────────────
// Customer Modal
// ─────────────────────────────────────────────
async function openModal(id) {
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';

  // Reset
  document.getElementById('modal-name').textContent = `Customer #${id}`;
  document.getElementById('modal-segment-badge').textContent = '...';

  const data = await apiFetch(`/customer/${id}`);
  if (!data) return;

  document.getElementById('modal-avatar').textContent = String(id).slice(-2);
  document.getElementById('modal-name').textContent   = `Customer #${id}`;
  document.getElementById('modal-segment-badge').textContent = data.segment_name ?? '—';

  document.getElementById('m-response').textContent = fmt_pct(data.response_prob);
  document.getElementById('m-churn').textContent    = fmt_pct(data.churn_prob);
  document.getElementById('m-ltv').textContent      = fmt_curr(data.predicted_ltv);
  document.getElementById('m-channel').textContent  = data.recommended_channel ?? '—';

  // Details grid
  const grid = document.getElementById('modal-details-grid');
  const EDU  = ['Basic','2n Cycle','Graduation','Master','PhD'];
  grid.innerHTML = [
    ['Age',         data.age],
    ['Income',      fmt_curr(data.income)],
    ['Education',   EDU[data.education] ?? '—'],
    ['Partnered',   data.is_partnered ? 'Yes' : 'No'],
    ['Children',    data.total_children],
    ['Recency',     data.recency + ' days'],
    ['Tenure',      Math.round(data.tenure_days/365) + ' years'],
    ['Campaigns',   data.campaign_history + '/5 accepted'],
    ['Complained',  data.complain ? 'Yes' : 'No'],
    ['Response Tier', tierBadge(data.response_tier ?? '—')],
    ['Churn Risk',    churnBadge(data.churn_risk ?? '—')],
    ['Top Product',   `<span class="badge badge-product">${data.top_product ?? '—'}</span>`],
  ].map(([lbl, val]) => `
    <div class="modal-detail">
      <div class="md-label">${lbl}</div>
      <div class="md-value">${val}</div>
    </div>`).join('');

  // Actual spend chart
  if (data.actual_spend) {
    if (modalSpendChart) modalSpendChart.destroy();
    const ctx = document.getElementById('modal-spend-chart').getContext('2d');
    const labels = Object.keys(data.actual_spend);
    const values = Object.values(data.actual_spend);
    modalSpendChart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: PALETTE, borderRadius: 5, borderSkipped:false }] },
      options: { plugins: { legend: { display: false } }, scales: {
        x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } }
      }}
    });
  }

  // Predicted spend chart
  const predSpend = {};
  ['wines','meat','fish','fruits','sweets','gold'].forEach(p => {
    const key = `pred_${p}`;
    if (data[key] != null) predSpend[p.charAt(0).toUpperCase()+p.slice(1)] = data[key];
  });
  if (Object.keys(predSpend).length) {
    if (modalPredChart) modalPredChart.destroy();
    const ctx2 = document.getElementById('modal-pred-chart').getContext('2d');
    modalPredChart = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: Object.keys(predSpend),
        datasets: [{ data: Object.values(predSpend), backgroundColor: PALETTE.map(c=>c+'aa'), borderRadius: 5, borderSkipped:false }]
      },
      options: { plugins: { legend: { display: false } }, scales: {
        x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } }
      }}
    });
  }
}

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
});
function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
  document.body.style.overflow = '';
}

// ─────────────────────────────────────────────
// Predict New Customer
// ─────────────────────────────────────────────
document.getElementById('btn-predict').addEventListener('click', async () => {
  const payload = {
    income:          parseFloat(document.getElementById('p-income').value),
    age:             parseInt(document.getElementById('p-age').value),
    education:       parseInt(document.getElementById('p-education').value),
    is_partnered:    parseInt(document.getElementById('p-partnered').value),
    has_children:    parseInt(document.getElementById('p-children').value) > 0 ? 1 : 0,
    total_children:  parseInt(document.getElementById('p-children').value),
    recency:         parseInt(document.getElementById('p-recency').value),
    tenure_days:     parseInt(document.getElementById('p-tenure').value),
    campaign_history:parseInt(document.getElementById('p-campaigns').value),
    num_web_visits:  parseInt(document.getElementById('p-webvisits').value),
    complain:        parseInt(document.getElementById('p-complain').value),
  };

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Running...';

  try {
    const res = await fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderPrediction(data);
  } catch(e) {
    console.error(e);
    alert('Prediction failed — is the backend running?');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Run ML Predictions';
  }
});

function renderPrediction(data) {
  document.querySelector('.predict-placeholder').style.display = 'none';
  const out = document.getElementById('prediction-output');
  out.style.display = 'block';

  document.getElementById('res-response-prob').textContent = fmt_pct(data.response_probability);
  document.getElementById('res-response-tier').textContent = data.response_tier ?? '—';
  document.getElementById('res-churn-prob').textContent    = fmt_pct(data.churn_probability);
  document.getElementById('res-churn-tier').textContent    = data.churn_risk ?? '—';
  document.getElementById('res-ltv').textContent           = fmt_curr(data.predicted_ltv);
  document.getElementById('res-channel').textContent       = data.recommended_channel ?? '—';
  document.getElementById('res-segment').textContent       = data.segment_name ?? '—';
  document.getElementById('res-top-product').textContent   = data.top_product ?? '—';

  // Affinity bars
  const barsEl = document.getElementById('res-affinity-bars');
  barsEl.innerHTML = '';
  if (data.predicted_spend) {
    const maxSpend = Math.max(...Object.values(data.predicted_spend));
    const ranking  = data.affinity_ranking || Object.keys(data.predicted_spend);
    ranking.forEach((prod, i) => {
      const amt = data.predicted_spend[prod] ?? 0;
      const pct = maxSpend > 0 ? (amt / maxSpend) * 100 : 0;
      const color = PALETTE[i % PALETTE.length];
      barsEl.innerHTML += `
        <div class="affinity-bar-row">
          <div class="affinity-bar-label">${prod}</div>
          <div class="affinity-bar-track">
            <div class="affinity-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="affinity-bar-amt">$${Math.round(amt)}</div>
        </div>`;
    });
  }

  // Strategy recommendation
  const resp  = data.response_probability ?? 0;
  const churn = data.churn_probability ?? 0;
  const ltv   = data.predicted_ltv ?? 0;
  const chan  = data.recommended_channel ?? 'Store';
  const prod  = data.top_product ?? 'this product';
  let strategy = '';
  if (resp >= 0.6 && churn < 0.35) {
    strategy = `<strong>High-value active customer.</strong> Send a <strong>${chan}</strong> campaign promoting <strong>${prod}</strong>. Predicted LTV of <strong>${fmt_curr(ltv)}</strong> makes this customer worth a premium offer.`;
  } else if (resp >= 0.6 && churn >= 0.35) {
    strategy = `<strong>Responsive but showing churn signals.</strong> Act quickly via <strong>${chan}</strong>. Lead with <strong>${prod}</strong> and include a loyalty reward to re-engage.`;
  } else if (resp < 0.3 && churn >= 0.65) {
    strategy = `<strong>High churn risk, low responsiveness.</strong> Consider a win-back sequence via <strong>${chan}</strong>. Keep offers simple — discount on <strong>${prod}</strong> with urgency.`;
  } else {
    strategy = `<strong>Moderate prospect.</strong> Reach via <strong>${chan}</strong> with a <strong>${prod}</strong>-focused message. Test with a low-cost touchpoint first.`;
  }
  document.getElementById('res-strategy').innerHTML = strategy;
}

// ─────────────────────────────────────────────
// Insights
// ─────────────────────────────────────────────
async function loadInsights() {
  const data = await apiFetch('/model-insights');
  if (!data) return;

  // Response feature importance
  if (data.feature_importances?.response) {
    renderFeatureChart('chart-feat-response', data.feature_importances.response, '#7c3aed');
  }
  if (data.feature_importances?.churn) {
    renderFeatureChart('chart-feat-churn', data.feature_importances.churn, '#ef4444');
  }
  if (data.feature_importances?.ltv) {
    renderFeatureChart('chart-feat-ltv', data.feature_importances.ltv, '#f59e0b');
  }

  // Metrics cards
  if (data.metrics) {
    const m = data.metrics;
    document.getElementById('metrics-response').innerHTML = [
      ['Total Customers',    fmt_int(m.total_customers)],
      ['Predicted Responders', fmt_int(m.predicted_responders)],
      ['High Churn Customers', fmt_int(m.high_churn_customers)],
      ['Avg Predicted LTV',   fmt_curr(m.avg_predicted_ltv)],
    ].map(([label, val]) => `
      <div class="metric-item">
        <span class="metric-label">${label}</span>
        <span class="metric-value">${val}</span>
      </div>`).join('');
  }
}

function renderFeatureChart(canvasId, feats, color) {
  destroyChart(canvasId);
  const top = [...feats].sort((a,b) => b.importance - a.importance).slice(0, 15).reverse();
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(f => f.feature),
      datasets: [{
        data: top.map(f => f.importance),
        backgroundColor: top.map((f, i) => {
          const alpha = 0.4 + (i / top.length) * 0.6;
          return color + Math.round(alpha * 255).toString(16).padStart(2,'0');
        }),
        borderRadius: 4, borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
}

// Insight tabs
document.querySelectorAll('.insight-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.insight-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.insight-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ─────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────
async function init() {
  const overview = await checkStatus();
  if (!overview) {
    document.getElementById('status-text').textContent = 'Backend offline — start server first';
    return;
  }
  // Load all overview data in parallel
  await Promise.all([
    loadOverview(),
    loadResponseDist(),
    loadSegmentChart(),
    loadProductChart(),
  ]);
  // Segment matrix depends on segments being loaded
  await loadSegmentMatrix();
  // Pre-load explorer table
  await loadTable(1);
}

// Navigate to explorer from nav loads table
document.getElementById('nav-explorer').addEventListener('click', () => {
  if (!document.getElementById('table-body').children.length ||
      document.getElementById('table-body').children[0].classList.contains('table-loading')) {
    loadTable(1);
  }
});

init();
