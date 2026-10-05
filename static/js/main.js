/* ═══════════════════════════════════════════════════════
   CardioSense AI — Main JavaScript
   ═══════════════════════════════════════════════════════ */

'use strict';

const API_BASE = '';   // same origin — Flask serves everything

// ── DOM refs ────────────────────────────────────────────
const form        = document.getElementById('predict-form');
const predictBtn  = document.getElementById('predict-btn');
const btnText     = predictBtn.querySelector('.btn-text');
const btnLoading  = predictBtn.querySelector('.btn-loading');

const resultPanel  = document.getElementById('result-panel');
const resultIdle   = document.getElementById('result-idle');
const resultOutput = document.getElementById('result-output');
const resultError  = document.getElementById('result-error');

const ringProgress  = document.getElementById('ring-progress');
const ringPct       = document.getElementById('ring-pct');
const riskRing      = document.getElementById('risk-ring');
const riskVerd      = document.getElementById('risk-verdict');
const verdictIcon   = document.getElementById('verdict-icon');
const verdictTitle  = document.getElementById('verdict-title');
const verdictSub    = document.getElementById('verdict-subtitle');
const resProbEl     = document.getElementById('res-prob');
const resModelEl    = document.getElementById('res-model');
const resultNoteEl  = document.getElementById('result-note');
const resultNoteText= document.getElementById('result-note-text');
const resetBtn      = document.getElementById('reset-btn');
const errorMsgEl    = document.getElementById('error-msg');

const statAcc  = document.getElementById('stat-acc');
const statAuc  = document.getElementById('stat-auc');
const modelsGrid = document.getElementById('models-grid');
const modelsNote = document.getElementById('models-note');
const bestFooter = document.getElementById('best-model-footer');

// Ring circumference: 2π × 52 ≈ 326.73
const RING_CIRC = 2 * Math.PI * 52;

// ── State helpers ───────────────────────────────────────
function showIdle()   { resultIdle.style.display='flex'; resultOutput.style.display='none'; resultError.style.display='none'; resultPanel.className='result-panel'; }
function showOutput() { resultIdle.style.display='none'; resultOutput.style.display='flex'; resultError.style.display='none'; }
function showError(msg) { resultIdle.style.display='none'; resultOutput.style.display='none'; resultError.style.display='flex'; errorMsgEl.textContent = msg; }

function setLoading(on) {
  btnText.style.display    = on ? 'none'  : 'flex';
  btnLoading.style.display = on ? 'flex'  : 'none';
  predictBtn.disabled = on;
}

// ── Animate ring ────────────────────────────────────────
function animateRing(pct, isHigh) {
  const color  = isHigh ? '#ef4444' : '#22c55e';
  const offset = RING_CIRC * (1 - pct / 100);
  ringProgress.style.stroke           = color;
  ringProgress.style.strokeDashoffset = RING_CIRC;  // reset
  ringPct.textContent = '0%';

  // Trigger reflow to restart transition
  void ringProgress.getBoundingClientRect();
  ringProgress.style.strokeDashoffset = offset;

  // Animate counter
  const duration = 1200;
  const start    = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    ringPct.textContent = Math.round(ease * pct) + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Render result ───────────────────────────────────────
function renderResult(data) {
  const prob   = data.probability;      // 0–100
  const isHigh = data.prediction === 1;

  // Panel border
  resultPanel.className = `result-panel risk-panel--${isHigh ? 'high' : 'low'}`;

  // Ring
  animateRing(prob, isHigh);

  // Verdict
  verdictIcon.textContent = isHigh ? '⚠️' : '✅';
  verdictIcon.className   = `verdict-icon verdict-icon--${isHigh ? 'high' : 'low'}`;
  verdictTitle.textContent = isHigh ? 'High Risk Detected' : 'Low Risk Detected';
  verdictSub.textContent   = isHigh
    ? 'Indicators suggest elevated cardiac risk.'
    : 'No significant cardiac risk indicators found.';

  // Meta
  resProbEl.textContent  = prob.toFixed(1) + '%';
  resModelEl.textContent = data.model_used;

  // Note
  resultNoteText.textContent = isHigh
    ? 'This result suggests a higher likelihood of heart disease. Please consult a cardiologist for a thorough evaluation.'
    : 'This result suggests a lower likelihood of heart disease. Maintain a healthy lifestyle and schedule regular check-ups.';

  showOutput();
}

// ── Form submit ─────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  setLoading(true);

  const fd   = new FormData(form);
  const body = {};
  fd.forEach((v, k) => { body[k] = parseFloat(v); });

  try {
    const res  = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || `Server error ${res.status}`);
    } else {
      renderResult(data);
      // Smooth scroll to result on mobile
      if (window.innerWidth < 900) {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  } catch (err) {
    showError('Could not reach the server. Make sure the backend is running on port 5000.');
  } finally {
    setLoading(false);
  }
});

// ── Reset ───────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  form.reset();
  showIdle();
  form.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

window.showIdle = showIdle;   // expose for inline onclick

// ── Load model info ─────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch(`${API_BASE}/api/model-info`);
    const data = await res.json();

    // Hero stats
    if (data.accuracy) statAcc.textContent = data.accuracy.toFixed(1) + '%';
    if (data.auc)      statAuc.textContent = data.auc.toFixed(1) + '%';

    // Footer
    if (bestFooter && data.model_name) {
      bestFooter.textContent = `Active model: ${data.model_name} · AUC ${data.auc?.toFixed(1)}%`;
    }

    // Model cards
    if (data.all_results && data.all_results.length) {
      modelsGrid.innerHTML = '';
      // Sort by AUC desc
      const sorted = [...data.all_results].sort((a, b) => b.auc - a.auc);
      sorted.forEach((m, i) => {
        const isBest = m.name === data.model_name;
        const card   = document.createElement('div');
        card.className = `model-card${isBest ? ' best-model' : ''}`;
        card.innerHTML = `
          <div class="model-name">
            ${m.name}
            ${isBest ? '<span class="best-badge">★ Best</span>' : ''}
          </div>
          <div class="model-metrics">
            <div class="metric-row">
              <span class="metric-label">Accuracy</span>
              <div class="metric-bar-wrap">
                <div class="metric-bar acc" style="width:0%" data-target="${m.accuracy}"></div>
              </div>
              <span class="metric-val">${m.accuracy.toFixed(1)}%</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">AUC-ROC</span>
              <div class="metric-bar-wrap">
                <div class="metric-bar auc" style="width:0%" data-target="${m.auc}"></div>
              </div>
              <span class="metric-val">${m.auc.toFixed(1)}%</span>
            </div>
          </div>`;
        modelsGrid.appendChild(card);
      });

      // Animate bars after DOM is rendered
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          document.querySelectorAll('.metric-bar').forEach(bar => {
            bar.style.width = bar.dataset.target + '%';
          });
        });
      });

      modelsNote.textContent = `Dataset: Cleveland UCI · ${data.all_results.length} models evaluated · Best selected by AUC-ROC`;
      modelsNote.style.display = 'block';
    }

  } catch (err) {
    // Backend not ready yet — retry after 2s
    setTimeout(loadModelInfo, 2000);
  }
}

// ── Select styling ──────────────────────────────────────
document.querySelectorAll('select').forEach(sel => {
  sel.addEventListener('change', () => {
    sel.classList.toggle('filled', sel.value !== '');
  });
});

// ── Init ────────────────────────────────────────────────
showIdle();
loadModelInfo();
