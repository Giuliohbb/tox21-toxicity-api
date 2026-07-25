import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.predict import InvalidSmilesError, predict

app = FastAPI()

ASSAY = "NR-AhR"
MODEL_NAME = "GCN"

INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tox21 NR-AhR predictor</title>
<style>
body { font-family: sans-serif; max-width: 480px; margin: 3rem auto; }
input { width: 60%; padding: 0.4rem; }
button { padding: 0.4rem 0.8rem; }
#result { margin-top: 1rem; font-weight: bold; }
</style>
</head>
<body>
<h1>Tox21 NR-AhR predictor</h1>
<input id="smiles" type="text" placeholder="Enter SMILES, e.g. CCO">
<button onclick="predict()">Predict</button>
<div id="result"></div>
<script>
async function predict() {
  const smiles = document.getElementById("smiles").value;
  const resultEl = document.getElementById("result");
  resultEl.textContent = "...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smiles }),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = "Error: " + data.detail;
      return;
    }
    resultEl.textContent = "Probability: " + data.probability.toFixed(4);
  } catch (err) {
    resultEl.textContent = "Error: request failed";
  }
}
</script>
</body>
</html>
"""


class PredictRequest(BaseModel):
    smiles: str


class PredictResponse(BaseModel):
    smiles: str
    probability: float
    assay: str
    model: str


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        probability = predict(request.smiles)
    except InvalidSmilesError:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles!r}")

    return PredictResponse(
        smiles=request.smiles,
        probability=probability,
        assay=ASSAY,
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
