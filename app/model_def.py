# Vendored from gnn-tox21-evaluation-protocols/src/models.py (+ featurization.py
# for AtomEncoder/ATOM_DIMS). Kept self-contained on purpose: this app does not
# depend on the research repo, only on the trained artifacts/model.pt state_dict.
# Only the GCN class and its atom encoder are included — GIN, GAT, factory
# helpers, and the training/sanity-check code from the original models.py were
# removed. No layer definition, dimension, or parameter name was changed, so
# artifacts/model.pt's state_dict still loads with strict=True.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

# data.x shape: [N_atoms, 9] — dtype long
ATOM_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]
# col 0: atomic_num (0-118)
# col 1: chirality  (9 tipos)
# col 2: degree     (0-10)
# col 3: formal_charge (-5 a +6 -> 12 valores)
# col 4: num_hs     (0-8)
# col 5: num_radical_electrons (0-4)
# col 6: hybridization (8 tipos)
# col 7: is_aromatic  (False/True -> 2)
# col 8: is_in_ring   (False/True -> 2)


class AtomEncoder(nn.Module):
    """
    Converte data.x (long, [N, 9]) em embeddings continuos ([N, hidden]).

    Mecanismo: um nn.Embedding por coluna categorica, todos com saida de
    dimensao `hidden`. A representacao final de cada atomo e a SOMA dos
    9 embeddings — estrategia identica a do OGB para molecular property
    prediction, que mantem o numero de parametros linear em `hidden`.

    Args:
        hidden (int): dimensao do vetor de saida por atomo.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, hidden) for dim in ATOM_DIMS]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.long, shape [N, 9]
        Returns:
            torch.float, shape [N, hidden]
        """
        # Soma os 9 embeddings coluna a coluna
        out = self.embeddings[0](x[:, 0])
        for i, emb in enumerate(self.embeddings[1:], start=1):
            out = out + emb(x[:, i])
        return out  # [N, hidden], float automaticamente


class _BaseMolecularGNN(nn.Module):
    """
    Base comum para classificadores moleculares com GNN.

    Mantem a interface pareada entre GCN, GIN e GAT: mesma featurizacao,
    mesma profundidade, mesma dimensao oculta, mesmo dropout, mesmo pooling
    e mesmo head de saida. As subclasses definem apenas o tipo de convolucao.

    Args:
        hidden: dimensao oculta dos atomos e grafos.
        num_layers: numero de camadas de message passing.
        dropout: probabilidade de dropout aplicada apos cada camada.
        pooling: "mean" ou "add" para agregacao grafo-level.
        out_channels: numero de logits por grafo. O pipeline atual usa 1.
    """

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
        out_channels: int = 1,
    ):
        super().__init__()

        hidden = int(hidden)
        num_layers = int(num_layers)
        out_channels = int(out_channels)

        if hidden <= 0:
            raise ValueError("hidden deve ser positivo.")
        if num_layers <= 0:
            raise ValueError("num_layers deve ser positivo.")
        if out_channels <= 0:
            raise ValueError("out_channels deve ser positivo.")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout deve estar no intervalo [0, 1].")
        if pooling not in {"mean", "add"}:
            raise ValueError("pooling deve ser 'mean' ou 'add'.")

        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.pooling = pooling
        self.out_channels = out_channels

        self.atom_encoder = AtomEncoder(hidden)
        self.convs = nn.ModuleList()
        self.readout = nn.Linear(hidden, out_channels)

    def _pool(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch_index)
        return global_add_pool(x, batch_index)

    @staticmethod
    def _batch_index(batch, x: torch.Tensor) -> torch.Tensor:
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            return x.new_zeros(x.size(0), dtype=torch.long)
        return batch_index

    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch/Data com x, edge_index e opcionalmente batch.

        Returns:
            logits [B, out_channels], sem sigmoid.
        """
        x = self.atom_encoder(batch.x)
        edge_index = batch.edge_index
        batch_index = self._batch_index(batch, x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = self._pool(x, batch_index)
        return self.readout(graph_emb)


class GCN(_BaseMolecularGNN):
    """
    Graph Convolutional Network baseada em GCNConv.

    Usa os hiperparametros comuns do projeto: hidden, num_layers e dropout.
    edge_attr e ignorado de proposito, pois a GCN vanilla usa somente a
    topologia do grafo e as features dos nos.
    """

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
        out_channels: int = 1,
    ):
        super().__init__(
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            out_channels=out_channels,
        )
        for _ in range(self.num_layers):
            self.convs.append(GCNConv(self.hidden, self.hidden))
