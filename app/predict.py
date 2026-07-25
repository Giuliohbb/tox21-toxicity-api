"""Loads artifacts/model.pt once at import time and exposes predict(smiles).

The checkpoint stores raw logits (see train.py: evaluate()/predict_scores()
in the research repo apply torch.sigmoid(logits) before computing AUROC/MCC).
metrics.json's mcc_threshold was chosen on those sigmoid outputs, so predict()
must return sigmoid(logit) to stay in the same [0, 1] probability space.
"""

from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from app.model_def import GCN

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.pt"


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


def _load_model():
    checkpoint = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = GCN(
        hidden=config["hidden"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


_MODEL = _load_model()


def predict(smiles: str) -> float:
    """Returns the predicted probability (in [0, 1]) that `smiles` is active
    in the NR-AhR assay.

    Raises:
        InvalidSmilesError: if `smiles` cannot be parsed into a molecule.
    """
    if Chem.MolFromSmiles(smiles) is None:
        raise InvalidSmilesError(f"Could not parse SMILES: {smiles!r}")

    data = from_smiles(smiles)
    if data.num_nodes == 0:
        raise InvalidSmilesError(f"Could not parse SMILES: {smiles!r}")

    batch = Batch.from_data_list([data])

    with torch.no_grad():
        logits = _MODEL(batch)
        prob = torch.sigmoid(logits)

    return float(prob.item())


if __name__ == "__main__":
    for smiles in ("CCO", "c1ccccc1"):
        print(f"{smiles!r} -> {predict(smiles)}")

    try:
        predict("not_a_molecule")
    except InvalidSmilesError as exc:
        print(f"'not_a_molecule' -> {exc}")
