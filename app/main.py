import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.predict import InvalidSmilesError, predict

app = FastAPI()

INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tox21 predictor</title>
<style>
:root {
  color-scheme: light;
  --page-plane:     #f9f9f7;
  --surface-1:      #fcfcfb;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --border-ui:      rgba(11,11,11,0.10);
  --track-bg:       #e1e0d9;
  --ramp-cool:      #2a78d6;
  --ramp-mid:       #c3c2b7;
  --ramp-warm:      #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page-plane:     #0d0d0d;
    --surface-1:      #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --border-ui:      rgba(255,255,255,0.10);
    --track-bg:       #2c2c2a;
    --ramp-cool:      #3987e5;
    --ramp-mid:       #383835;
    --ramp-warm:      #d95926;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--page-plane);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}

.wrap {
  max-width: 960px;
  margin: 0 auto;
  padding: 1.25rem 1rem 2rem;
}

h1 { font-size: 1.4rem; }

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1rem 0;
}

#smiles {
  flex: 1 1 220px;
  min-height: 44px;
  font-size: 16px;
  padding: 0 0.75rem;
  border: 1px solid var(--border-ui);
  border-radius: 8px;
  background: var(--surface-1);
  color: var(--text-primary);
}

#predictBtn {
  min-height: 44px;
  padding: 0 1.25rem;
  font-size: 16px;
  border: 1px solid var(--border-ui);
  border-radius: 8px;
  background: var(--surface-1);
  color: var(--text-primary);
}

#status {
  min-height: 1.4em;
  font-weight: bold;
}

.assays {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

@media (min-width: 700px) {
  .assays { flex-direction: row; }
  .assay { flex: 1 1 0; }
}

.assay {
  background: var(--surface-1);
  border: 1px solid var(--border-ui);
  border-radius: 12px;
  padding: 0.85rem 0.75rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 0.4rem;
}

.assay-name { font-weight: 600; }

.cyl-track {
  position: relative;
  width: 60px;
  height: 160px;
  background: var(--track-bg);
  border: 1px solid var(--border-ui);
  border-radius: 14px;
  overflow: hidden;
}

.cyl-fill {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
}

.pct-text { font-size: 0.9rem; }
.prob-text { font-size: 0.75rem; color: var(--text-secondary); }
.auroc-text { font-size: 0.75rem; color: var(--text-muted); }
.low-rel-caption { font-size: 0.7rem; font-style: italic; color: var(--text-muted); }

.viz-note { font-size: 0.75rem; color: var(--text-muted); margin-top: 1rem; }
</style>
</head>
<body>
<div class="wrap">
<h1>Tox21 predictor</h1>
<div class="controls">
<input id="smiles" type="text" list="smilesExamples" placeholder="Enter SMILES, e.g. CCO">
<button id="predictBtn" onclick="predict()">Predict</button>
</div>
<datalist id="smilesExamples">
<option value="CCO">Ethanol (train)</option>
<option value="CC(C)Cc1ccc(C(C)C(=O)O)cc1">Ibuprofen (train)</option>
<option value="CC(=O)Oc1ccccc1C(=O)O">Aspirin (train)</option>
<option value="Cn1c(=O)c2c(ncn2C)n(C)c1=O">Caffeine (test)</option>
<option value="c1ccc2ccccc2c1">Naphthalene (train)</option>
<option value="c1ccc2c(c1)ccc1ccccc12">Phenanthrene (train)</option>
<option value="Clc1cc2c(cc1Cl)Oc1cc(Cl)c(Cl)cc1O2">TCDD / dioxin (train)</option>
<option value="c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34">Benzo[a]pyrene (train)</option>
</datalist>
<div id="status"></div>
<div id="assays" class="assays"></div>
<p id="vizNote" class="viz-note"></p>
</div>
<script>
function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(rgb) {
  return "#" + rgb.map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("");
}

function mixHex(hexA, hexB, t) {
  const a = hexToRgb(hexA), b = hexToRgb(hexB);
  return rgbToHex(a.map((v, i) => v + (b[i] - v) * t));
}

function toGrayHex(hex) {
  const [r, g, b] = hexToRgb(hex);
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return rgbToHex([y, y, y]);
}

function desaturateHex(hex, amount) {
  return mixHex(hex, toGrayHex(hex), amount);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function rampColor(t) {
  t = Math.max(0, Math.min(1, t));
  const cool = cssVar("--ramp-cool");
  const mid = cssVar("--ramp-mid");
  const warm = cssVar("--ramp-warm");
  return t <= 0.5 ? mixHex(cool, mid, t / 0.5) : mixHex(mid, warm, (t - 0.5) / 0.5);
}

function reliabilityTier(auroc) {
  if (auroc >= 0.80) return "high";
  if (auroc >= 0.70) return "mid";
  return "low";
}

function renderAssays(results) {
  const assaysEl = document.getElementById("assays");
  const noteEl = document.getElementById("vizNote");
  assaysEl.innerHTML = "";

  for (const r of results) {
    const tier = reliabilityTier(r.auroc);
    const base = rampColor(r.percentile);
    let fillColor = base;
    let cylOpacity = 1;
    if (tier === "mid") fillColor = desaturateHex(base, 0.35);
    if (tier === "low") { fillColor = desaturateHex(base, 0.9); cylOpacity = 0.5; }

    const panel = document.createElement("div");
    panel.className = "assay";

    const name = document.createElement("div");
    name.className = "assay-name";
    name.textContent = r.assay;

    const track = document.createElement("div");
    track.className = "cyl-track";
    track.style.opacity = cylOpacity;
    const fill = document.createElement("div");
    fill.className = "cyl-fill";
    fill.style.height = (r.percentile * 100) + "%";
    fill.style.backgroundColor = fillColor;
    track.appendChild(fill);

    const pctText = document.createElement("div");
    pctText.className = "pct-text";
    pctText.textContent = "higher than " + Math.round(r.percentile * 100) + "% of training compounds";

    const probText = document.createElement("div");
    probText.className = "prob-text";
    probText.textContent = "probability " + r.probability.toFixed(4);

    const aurocText = document.createElement("div");
    aurocText.className = "auroc-text";
    aurocText.textContent = "AUROC " + r.auroc.toFixed(3);

    panel.append(name, track, pctText, probText, aurocText);

    if (tier === "low") {
      const caption = document.createElement("div");
      caption.className = "low-rel-caption";
      caption.textContent = "AUROC " + r.auroc.toFixed(3) + " — low reliability, shown for comparison";
      panel.appendChild(caption);
    }

    assaysEl.appendChild(panel);
  }

  noteEl.textContent = "Fill shows rank against the training distribution, not calibrated risk.";
}

async function predict() {
  const smiles = document.getElementById("smiles").value;
  const btn = document.getElementById("predictBtn");
  const statusEl = document.getElementById("status");
  const assaysEl = document.getElementById("assays");
  const noteEl = document.getElementById("vizNote");

  btn.disabled = true;
  statusEl.textContent = "Predicting…";
  assaysEl.innerHTML = "";
  noteEl.textContent = "";

  const coldStartTimer = setTimeout(() => {
    statusEl.textContent = "Cold start — the service scales to zero when idle, this can take ~30s.";
  }, 3000);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smiles }),
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = "Error: " + data.detail;
      return;
    }

    statusEl.textContent = "";
    renderAssays(data.results);
  } catch (err) {
    statusEl.textContent = "Error: request failed";
  } finally {
    clearTimeout(coldStartTimer);
    btn.disabled = false;
  }
}

document.getElementById("smiles").addEventListener("keydown", (event) => {
  if (event.key === "Enter") predict();
});
</script>
</body>
</html>
"""


class PredictRequest(BaseModel):
    smiles: str


class AssayResult(BaseModel):
    assay: str
    probability: float
    percentile: float
    auroc: float
    auprc: float


class PredictResponse(BaseModel):
    smiles: str
    results: list[AssayResult]


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        results = predict(request.smiles)
    except InvalidSmilesError:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles!r}")

    return PredictResponse(smiles=request.smiles, results=results)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
