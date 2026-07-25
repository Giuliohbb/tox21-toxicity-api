# tox21-toxicity-api

Molecular toxicity prediction API. Takes a molecule as SMILES and returns the
predicted probability for the **NR-AhR** assay of Tox21, using a Graph Neural
Network (GCN).

🚧 **Work in progress** — the public link goes here once the deployment is live.

## Background

The model comes from a comparative study on evaluation protocols for molecular
toxicity prediction (random vs. scaffold vs. UMAP split), covering six models and
four Tox21 assays:

**https://github.com/Giuliohbb/gnn-tox21-evaluation-protocols**

That study was a group project for a Deep Neural Networks course, with Giulio
Henrique Borges Basso(me), João Arthur Xavier Marques, Leonardo Lima de Oliveira, Livia
Maria Santos Rocha and Maria Eduarda Fernandes de Souza. Its documentation is in
Portuguese.

**This repository** covers only the work of taking that model to production,
serving, containerization and deployment, done individually by Giulio Henrique Borges Basso.

The model definition in `app/model_def.py` is a copy from the research repository.
The duplication is intentional: the serving artifact must be self-contained and
should not break due to changes upstream.

## Disclaimer

Portfolio project. Predictions are **not validated for clinical, regulatory or
safety use**. A toxicity model at this performance level is a screening
prioritization tool, not a verdict.

## License

MIT — see [LICENSE](LICENSE).