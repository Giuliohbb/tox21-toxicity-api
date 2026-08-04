"""Loads all four Tox21 assay artifacts once at import time into a dict keyed
by task name (artifacts/model_<TASK>.pt + artifacts/metrics_<TASK>.json).

The checkpoints store raw logits (see train.py: evaluate()/predict_scores()
in the research repo apply torch.sigmoid(logits) before computing AUROC/MCC
and before scoring the training set into train_scores), so predict() must
return sigmoid(logit) to stay in the same [0, 1] space as train_scores.

percentile is the fraction of a task's train_scores strictly below the
molecule's score for that task, via bisect on the pre-sorted list.
"""

import bisect
import json
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from app.model_def import GCN

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

ASSAYS = ["NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"]


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


def _load_assay(task: str) -> dict:
    checkpoint = torch.load(
        ARTIFACTS_DIR / f"model_{task}.pt", map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    model = GCN(
        hidden=config["hidden"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    with open(ARTIFACTS_DIR / f"metrics_{task}.json") as f:
        metrics = json.load(f)

    return {
        "model": model,
        "train_scores": sorted(checkpoint["train_scores"]),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
    }


MODELS = {task: _load_assay(task) for task in ASSAYS}


def _to_graph(smiles: str):
    if Chem.MolFromSmiles(smiles) is None:
        raise InvalidSmilesError(f"Could not parse SMILES: {smiles!r}")

    data = from_smiles(smiles)
    if data.num_nodes == 0:
        raise InvalidSmilesError(f"Could not parse SMILES: {smiles!r}")

    return data


def predict(smiles: str) -> list[dict]:
    """Returns one result dict per Tox21 assay for `smiles`:
    {"assay", "probability", "percentile", "auroc", "auprc"}.

    Validation happens once, up front, for the whole molecule — not per
    assay — so an unparseable SMILES fails before any model runs.

    Raises:
        InvalidSmilesError: if `smiles` cannot be parsed into a molecule.
    """
    data = _to_graph(smiles)
    batch = Batch.from_data_list([data])

    results = []
    for task in ASSAYS:
        entry = MODELS[task]

        with torch.no_grad():
            logits = entry["model"](batch)
            probability = float(torch.sigmoid(logits).item())

        train_scores = entry["train_scores"]
        percentile = bisect.bisect_left(train_scores, probability) / len(train_scores)

        results.append(
            {
                "assay": task,
                "probability": probability,
                "percentile": percentile,
                "auroc": entry["auroc"],
                "auprc": entry["auprc"],
            }
        )

    return results


if __name__ == "__main__":
    for smiles in ("CCO", "c1ccccc1"):
        print(f"{smiles!r} -> {predict(smiles)}")

    try:
        predict("not_a_molecule")
    except InvalidSmilesError as exc:
        print(f"'not_a_molecule' -> {exc}")
