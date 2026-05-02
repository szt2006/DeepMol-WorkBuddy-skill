# DeepMol Skill

> A [WorkBuddy](https://www.codebuddy.cn/) skill that wraps [DeepMol](https://github.com/BioSystemsUM/DeepMol) into an AI-assisted workflow for computational chemistry and drug discovery.

DeepMol is a Python-based machine learning and deep learning framework for drug discovery, built on RDKit, TensorFlow/Keras, PyTorch, scikit-learn, and DeepChem. **This repository packages DeepMol's capabilities as a reusable AI skill** — giving AI assistants the procedural knowledge to run end-to-end molecular ML workflows without guessing at APIs or best practices.

## What's Inside

```
deepmol-skill/
├── SKILL.md                          # Main skill guide (11-step molecular ML workflow)
├── LICENSE                           # MIT License
├── README.md                         # This file
├── references/
│   └── api_reference.md              # Full API reference for 14+ DeepMol modules
└── scripts/
    ├── check_install.py              # One-click environment verification
    ├── tutorial.py                   # Guided beginner tutorial (built-in demo data)
    ├── qsar_pipeline.py              # CLI: end-to-end QSAR/QSPR pipeline
    ├── batch_featurization.py        # CLI: batch molecular featurization
    └── admet_prediction.py           # CLI: ADMET property prediction (pre-trained models)
```

## Verified On

DeepMol 1.2.1, Python 3.14, Windows 11 (May 2026). All code in this skill has been run and verified against a live installation. API documentation reflects actual behavior, not guesswork from READMEs.

## What This Skill Teaches AI Assistants

| Capability | What the AI learns |
|---|---|
| **Data Loading** | CSV (SMILES) and SDF (3D) with sharded loading for large datasets |
| **Compound Standardization** | Basic, Custom Heavy, and ChEMBL standardizers |
| **Molecular Featurization** | Morgan, MACCS, RDK, AtomPair, Layered, Mol2Vec, DeepChem featurizers |
| **Feature Selection** | LowVariance, KBest, Percentile, RFE, Importance-based |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP, KMeans clustering |
| **Data Splitting** | Random, Stratified, K-fold, Scaffold-based |
| **Model Building** | scikit-learn, Keras/TensorFlow, and DeepChem backends |
| **Hyperparameter Tuning** | Grid & Randomized search with cross-validation |
| **Explainability** | SHAP values (beeswarm, waterfall, feature plots, bit-on-molecule visualization) |
| **Imbalanced Data** | SMOTE, SMOTEENN, SMOTETomek, ClusterCentroids |
| **Pipelines** | End-to-end prediction + save/load for reproducibility |
| **AutoML** | Optuna-powered pipeline step optimization |

## Prerequisites

**Python >= 3.13 note:** The official `pip install "deepmol[all]"` fails due to scikeras dependency. Use the verified step-by-step install from SKILL.md instead.

```bash
# Core + sklearn-first approach (works on Python 3.13+)
pip install deepmol --no-deps
pip install scikit-learn
pip install rdkit seaborn pillow h5py imbalanced-learn chembl_structure_pipeline \
    graph-part kneed shap umap-learn dill boruta ipython pandas biosynfoni \
    cached_property timeout_decorator matplotlib networkx plotly transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install optuna

# Verify
python scripts/check_install.py
```

## Quick Start After Install

```bash
# Guided tutorial (no data needed)
python scripts/tutorial.py

# Verify environment
python scripts/check_install.py
```

## Standalone Script Usage

The scripts in `scripts/` can also be used directly without WorkBuddy:

### Full QSAR Pipeline

```bash
python scripts/qsar_pipeline.py data.csv \
  --smiles-col SMILES --label-col pIC50 \
  --task regression --model rf --featurizer morgan --shap
```

### Batch Featurization

```bash
python scripts/batch_featurization.py library.csv \
  --smiles-col SMILES --featurizer morgan --output features.csv
```

### ADMET Prediction

```bash
python scripts/admet_prediction.py compounds.csv \
  --models BBBP CYP2D6 HIA Solubility --output predictions.csv
```

## How to Use as a WorkBuddy Skill

1. Copy this repository into `~/.workbuddy/skills/deepmol/`
2. Or install via WorkBuddy skill marketplace (if published)
3. Start a conversation with phrases like:
   - "Train a QSAR model on this dataset"
   - "Generate Morgan fingerprints for these molecules"
   - "Predict ADMET properties for my compound library"
   - "Optimize a classification pipeline with Optuna"

## References

- **DeepMol Paper:** Correia, J., Capela, J., & Rocha, M. (2024). *DeepMol: An Automated Machine and Deep Learning Framework for Computational Chemistry.* Journal of Cheminformatics, 16, 136. [DOI: 10.1186/s13321-024-00937-7](https://doi.org/10.1186/s13321-024-00937-7)
- **DeepMol GitHub:** https://github.com/BioSystemsUM/DeepMol
- **DeepMol Case Studies:** https://github.com/BioSystemsUM/deepmol_case_studies
- **WorkBuddy Docs:** https://www.codebuddy.cn/docs/workbuddy/Overview

## License

MIT — see [LICENSE](LICENSE) for details.

---

## 中文说明

### 这是什么？

这是一个 [WorkBuddy](https://www.codebuddy.cn/) 的 AI skill，将 [DeepMol](https://github.com/BioSystemsUM/DeepMol) 计算化学/药物发现框架的能力封装为 AI 助手可用的知识包。

### 包含内容

- **SKILL.md**：11 步完整分子机器学习工作流指南（从数据加载到管线优化）
- **references/api_reference.md**：DeepMol 所有模块的详细 API 参考
- **scripts/**：三个可直接使用的 CLI 脚本（QSAR 管线、批量特征化、ADMET 预测）

### 如何使用

将本仓库复制到 `~/.workbuddy/skills/deepmol/`，然后在 WorkBuddy 中直接说"帮我做 QSAR 建模"即可自动触发。
