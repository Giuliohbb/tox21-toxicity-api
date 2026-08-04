# tox21-toxicity-api

Molecular toxicity prediction served as a containerized API. Paste a molecule as
SMILES and get its predicted activity across **four Tox21 assays**, from Graph
Neural Networks trained under a leave-cluster-out evaluation protocol.

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
{
  "smiles": "CCO",
  "results": [
    {"assay": "NR-AhR", "probability": 0.0049, "percentile": 0.108,
     "auroc": 0.868, "auprc": 0.534},
    {"assay": "NR-ER",  "probability": 0.0107, "percentile": 0.011, ...}
  ]
}
```

| Endpoint | |
|---|---|
| `GET /` | Web UI with example molecules and per-assay visualization |
| `POST /predict` | `{"smiles": str}` → four assay results. Invalid SMILES → `400` |
| `GET /health` | Liveness check |

---

## The four assays

| Assay | Target | AUROC | AUPRC | Test set |
|---|---|---|---|---|
| **NR-AhR** | Aryl hydrocarbon receptor | **0.868** | 0.534 | 652 |
| SR-MMP | Mitochondrial membrane potential | 0.726 | 0.328 | 558 |
| NR-ER | Estrogen receptor | 0.686 | 0.260 | 685 |
| SR-ARE | Antioxidant response element | 0.595 | 0.325 | 554 |

**SR-ARE at 0.595 is barely above chance**, and the interface says so — assays
below 0.70 AUROC are rendered desaturated with an explicit low-reliability
caption rather than hidden or shown as equals. A dashboard that renders four
identically confident predictions, one of which is a coin flip, would be
misleading in exactly the way this project argues against.

The spread across assays is itself a finding of the underlying study: model
performance under a rigorous evaluation protocol is strongly assay-dependent.

**Every one of the four served models was verified end-to-end against its own
held-out test set** — the API was queried once per test molecule and the returned
probabilities reproduce each artifact's reported AUROC and AUPRC exactly. The
serving path is proven identical to the evaluated model, not assumed to be.
Verification script: `scripts/eval_api.py` in the research repo.

---

## How to read the output

The interface shows, per assay, a cylinder whose fill is the molecule's
**percentile rank against that assay's training distribution** — "higher than 94%
of training compounds" — not its raw probability.

This is deliberate. The models are **uncalibrated**: a predicted 0.80 does not
mean 80% of such molecules are active. Each assay also has its own decision
threshold and its own base rate, so the same probability means different things
across the four. Percentile rank is comparable across assays and expresses what
the model is actually for — **prioritizing which compounds to test first**, not
issuing a verdict on any single one.

Raw probabilities are still returned by the API and shown in smaller text.

These models predict activity in **specific assays**. A molecule can be inactive
in all four and still be harmful by another mechanism. This is not a general
toxicity score.

---

## Why the evaluation is unusual

Chemical datasets contain analog series — molecules differing by a single atom.
Split them randomly and near-identical structures land on both sides of the
train/test boundary. The model interpolates within known chemistry and scores
well, while virtual screening's actual job is extrapolating to novel chemistry.
This is **structural data leakage**, and it makes random-split numbers
systematically optimistic.

These models are trained and evaluated under a **UMAP split**
([Guo et al., 2024](https://doi.org/10.48550/arXiv.2406.00873)): ECFP
fingerprints → UMAP projection under Jaccard distance → K-Means clustering →
whole clusters held out. Test molecules come from regions of chemical space the
model never saw — not merely different scaffolds.

For NR-AhR, the same architecture under the three protocols:

| Protocol | AUROC (mean ± std, 5 seeds) |
|---|---|
| Random | 0.885 ± 0.011 |
| Scaffold | 0.853 ± 0.012 |
| UMAP | 0.877 ± 0.002 |

Note that here the scaffold split is *harder* than the UMAP split. That inversion
is one of the study's findings: nominal protocol rigor does not reliably predict
actual difficulty.

**The served NR-AhR artifact scores 0.868**, slightly below the 0.877 five-seed
mean. It is a separate training run — same configuration, same partition,
different effective RNG state at initialization. It was not re-run to match: the
number reported is the number the deployed model produces.

---

## Provenance

Models and methodology come from a comparative study of evaluation protocols —
6 models × 3 splits × 4 assays, ~4,680 training runs:

**https://github.com/Giuliohbb/gnn-tox21-evaluation-protocols**

That study was a group project for a Deep Neural Networks course, with Giulio
Henrique Borges Basso, João Arthur Xavier Marques, Leonardo Lima de Oliveira,
Livia Maria Santos Rocha and Maria Eduarda Fernandes de Souza. Its documentation
is in Portuguese.

**This repository** covers only taking those models to production — serving,
containerization and deployment — done individually by Giulio Basso.

The weights in `artifacts/` are produced by `scripts/export_model.py` in the
research repo, which reads each assay's winning hyperparameter configuration,
rebuilds the identical partition, trains once, and writes the weights and metrics
in the same run. Every number above describes those specific artifacts.

**GCN is served for all four assays**, even where the study found a fingerprint
baseline competitive. This is a deliberate consistency choice: serving a
different architecture per assay would mean a second inference path and a larger
dependency surface for a marginal gain.

---

## Architecture

```
SMILES → RDKit parse → PyG graph (atoms=nodes, bonds=edges)
       → 2-layer GCN (hidden=256, dropout=0.5) → mean pool → sigmoid
       → ×4 assays, from one parsed graph
```

Each model is ~177k parameters, selected by random search (20 configs × 3 seeds)
on validation AUROC, with search seeds disjoint from evaluation seeds. All four
artifacts load once at startup.

Inference runs sequentially. Four small GCNs on one molecule is a few
milliseconds of compute against a ~150ms network round trip — parallelizing it
would add complexity to 3% of the latency.

### Notes on the serving path

**Input validation is two-stage, and runs once before any model.**
`torch_geometric.utils.from_smiles` does not raise on invalid input — RDKit falls
back to parsing an empty string and returns a graph with zero atoms, which then
breaks `global_mean_pool`. The training pipeline never hit this because
`MoleculeNet.process()` drops empty graphs upstream. So the API checks
`Chem.MolFromSmiles` for `None` *and* `num_nodes == 0` before anything reaches a
tensor.

**The model definition is vendored** into `app/model_def.py` rather than imported
from the research repo. Deliberate duplication: a serving artifact must be
self-contained and must not break when the research code is refactored.
`load_state_dict(strict=True)` enforces that the copy stayed faithful.

**Dependencies are pinned to the exact environment that produced the weights** —
rdkit especially, since it performs featurization. A patch-level change to any
atom property encoding would shift the input tensors, raise no error, and
silently change predictions.

**Docker layers are ordered by change frequency.** Torch, the dominant cost of
the image, installs from the CPU-only index in its own layer before application
code, so editing the UI doesn't re-download it.

**Cloud Run runs with `--min-instances 0` and `--cpu-boost`.** Scale-to-zero
means near-zero cost at the price of a cold start — the right trade for a demo,
the wrong one for production traffic. CPU boost is the highest-leverage
mitigation, since importing torch is single-threaded and CPU-bound. Images are
tagged immutably (`:v1`…`:v6`) so any revision can be rolled back to known bytes.

---

## Limitations

- **No applicability domain.** The models return confident predictions for
  molecules from chemical space they have never seen — and with four assays,
  that is four times the overclaim. Flagging low max-Tanimoto similarity to the
  training set is the highest-priority improvement.
- **SR-ARE is not usable.** 0.595 AUROC. It is shown for comparison and marked
  as unreliable.
- **One seed per assay.** No uncertainty is reported.
- **Uncalibrated.** Percentile rank is shown precisely because raw probabilities
  are not calibrated risk.
- **Several example molecules are in the training set**, and are labelled as
  such. They demonstrate the interface, not generalization.
- **Disconnected inputs return a number.** Mean pooling ignores connectivity, so
  `[Na+].[Cl-]` — two atoms, zero bonds — produces output. It is not meaningful.
- **No automated tests or CI.**

**Not validated for clinical, regulatory or safety use.** Models at this
performance level are screening prioritization tools, not verdicts.

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