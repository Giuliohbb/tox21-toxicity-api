# tox21-toxicity-api

Molecular toxicity prediction served as a containerized API. Paste a molecule as
SMILES, get the predicted probability that it activates the **NR-AhR** assay of
Tox21, from a Graph Neural Network trained under a leave-cluster-out evaluation
protocol.

**Live: https://tox21-api-399156159770.us-central1.run.app**

> The service scales to zero when idle. Warm requests return in ~0.3s; the first
> request after a period of inactivity takes ~8s while the container boots and
> imports PyTorch.

---

## What it does

```bash
curl -X POST https://tox21-api-399156159770.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO"}'
```

```json
{"smiles": "CCO", "probability": 0.0049, "assay": "NR-AhR", "model": "GCN"}
```

| Endpoint | |
|---|---|
| `GET /` | Minimal web UI |
| `POST /predict` | `{"smiles": str}` → probability. Invalid SMILES → `400` |
| `GET /health` | Liveness check |

---

## What the number means

The model predicts activity in **one assay**: the aryl hydrocarbon receptor. A
molecule can be inactive here and harmful by another mechanism. This is not a
general toxicity score.

It is also a **ranking** model, not a classifier. The intended use is
prioritizing which compounds to test first, not labelling any single molecule.

**Performance of the exact artifact being served** (`artifacts/metrics.json`,
652 held-out molecules, 14.0% active):

| Metric | Value | |
|---|---|---|
| AUROC | 0.868 | |
| **AUPRC** | **0.534** | vs. 0.140 base rate — 3.8× |
| EF@5% | 4.56 | Top 5% by score yields 4.6× more actives than random |
| EF@1% | 4.09 | Top 1% is 7 molecules; small sample, read with caution |
| MCC | 0.418 | at threshold 0.572, selected on validation |
| BEDROC (α=20) | 0.622 | |

The AUROC/AUPRC gap is the point. AUROC 0.868 reads as nearly solved; AUPRC
0.534 against a 0.140 base rate describes what the model actually does on the
minority class. Under 14% positives, AUROC's false-positive-rate denominator is
dominated by negatives, so it stays high even when flagged compounds are mostly
wrong. Accuracy is not reported: predicting "inactive" for everything scores 86%.

**The deployed endpoint was evaluated against all 652 held-out test molecules and
reproduces the artifact's reported AUROC (0.8680) and AUPRC (0.5336)** — the
serving path is verified identical to the evaluated model, not assumed to be.
Verification script: `scripts/eval_api.py` in the research repo.

---

## Why the evaluation is unusual

Chemical datasets contain analog series — molecules differing by a single atom.
Split them randomly and near-identical structures land on both sides of the
train/test boundary. The model interpolates within known chemistry and scores
well, while virtual screening's actual job is extrapolating to novel chemistry.
This is **structural data leakage**, and it makes random-split numbers
systematically optimistic.

This model is trained and evaluated under a **UMAP split**
([Guo et al., 2024](https://doi.org/10.48550/arXiv.2406.00873)): ECFP
fingerprints → UMAP projection under Jaccard distance → K-Means clustering →
whole clusters held out. Test molecules come from regions of chemical space the
model never saw — not merely different scaffolds.

For context, the same architecture on this assay:

| Protocol | AUROC (mean ± std, 5 seeds) |
|---|---|
| Random | 0.885 ± 0.011 |
| Scaffold | 0.853 ± 0.012 |
| UMAP | 0.877 ± 0.002 |

Note that on NR-AhR the scaffold split is *harder* than the UMAP split. That
inversion is one of the findings of the underlying study: nominal protocol rigor
does not reliably predict actual difficulty, which turns out to be
assay-dependent.

**The served artifact scores 0.868**, slightly below the 0.877 five-seed mean. It
is a separate training run — same configuration, same partition, different
effective RNG state at initialization. It was not re-run to match: the number
reported is the number the deployed model produces.

---

## Provenance

Model and methodology come from a comparative study of evaluation protocols —
6 models × 3 splits × 4 assays, ~4,680 training runs:

**https://github.com/Giuliohbb/gnn-tox21-evaluation-protocols**

That study was a group project for a Deep Neural Networks course, with Giulio
Henrique Borges Basso, João Arthur Xavier Marques, Leonardo Lima de Oliveira,
Livia Maria Santos Rocha and Maria Eduarda Fernandes de Souza. Its documentation
is in Portuguese.

**This repository** covers only taking that model to production — serving,
containerization and deployment — done individually by Giulio Basso.

The weights in `artifacts/model.pt` are produced by `scripts/export_model.py` in
the research repo, which reads the winning hyperparameter configuration, rebuilds
the identical partition, trains once, and writes the weights and `metrics.json`
in the same run. Every number above describes that specific artifact.

---

## Architecture

```
SMILES → RDKit parse → PyG graph (atoms=nodes, bonds=edges)
       → 2-layer GCN (hidden=256, dropout=0.5) → mean pool → sigmoid
```

177,153 parameters. Config:
`{"hidden": 256, "num_layers": 2, "dropout": 0.5, "lr": 0.001, "weight_decay": 0.0}`,
selected by random search (20 configs × 3 seeds) on validation AUROC, with search
seeds disjoint from evaluation seeds.

### Notes on the serving path

**Input validation is two-stage.** `torch_geometric.utils.from_smiles` does not
raise on invalid input — RDKit falls back to parsing an empty string and returns a
graph with zero atoms, which then breaks `global_mean_pool`. The training pipeline
never hit this because `MoleculeNet.process()` drops empty graphs upstream. So the
API checks `Chem.MolFromSmiles` for `None` *and* `num_nodes == 0` before anything
reaches a tensor.

**The model definition is vendored** into `app/model_def.py` rather than imported
from the research repo. Deliberate duplication: a serving artifact must be
self-contained and must not break when the research code is refactored.
`load_state_dict(strict=True)` enforces that the copy stayed faithful.

**Dependencies are pinned to the exact environment that produced the weights** —
rdkit especially, since it performs featurization. A patch-level change to any
atom property encoding would shift the input tensors, raise no error, and silently
change predictions.

**Docker layers are ordered by change frequency.** Torch (822 MB of a 1.23 GB
image) installs from the CPU-only index in its own layer, before application code,
so editing `main.py` doesn't re-download it.

**Cloud Run runs with `--min-instances 0` and `--cpu-boost`.** Scale-to-zero means
near-zero cost at the price of a cold start — the right trade for a demo, the wrong
one for production traffic. CPU boost is the highest-leverage mitigation, since
importing torch is single-threaded and CPU-bound. Images are tagged immutably
(`:v1`, `:v2`) so any revision can be rolled back to known bytes.

---

## Limitations

- **No applicability domain.** The model returns confident predictions for
  molecules from chemical space it has never seen. Flagging low max-Tanimoto
  similarity to the training set is the highest-priority improvement.
- **One assay, one seed.** No uncertainty is reported. The study's mean ± std is
  context, not a property of this artifact.
- **Probability, not a verdict.** The MCC threshold was selected on validation
  (9.6% positives) and applied to test (14.0%). It transfers poorly, so no
  binary label is exposed.
- **Uncalibrated.** A predicted 0.80 does not mean 80% of such molecules are
  active.
- **Disconnected inputs return a number.** Mean pooling ignores connectivity, so
  `[Na+].[Cl-]` — two atoms, zero bonds — produces output. It is not meaningful.
- **No automated tests or CI.** Deliberate scope for a weekend build.

**Not validated for clinical, regulatory or safety use.** A model at this
performance level is a screening prioritization tool, not a verdict.

---

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080
```

```bash
docker build -t tox21-api .
docker run -p 8080:8080 tox21-api
```

---

## License

MIT — see [LICENSE](LICENSE).