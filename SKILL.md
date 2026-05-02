---
name: deepmol
description: "This skill enables AI-powered computational chemistry and drug discovery workflows using the DeepMol framework. Use when the user asks to load molecular data (SMILES/SDF), standardize compounds, featurize molecules (Morgan/MACCS/mol2vec fingerprints), perform feature selection, train and evaluate ML/DL models (scikit-learn, Keras/TensorFlow, DeepChem), optimize hyperparameters, explain predictions with SHAP, build end-to-end pipelines, or run automated pipeline optimization with Optuna. Also use for ADMET property prediction, QSAR/QSPR modeling, virtual screening, and molecular property prediction using pre-trained DeepMol models. Trigger keywords: DeepMol, SMILES, molecular, fingerprint, QSAR, ADMET, drug discovery, cheminformatics, compound, RDKit, Morgan fingerprint, MACCS keys, Mol2Vec, SHAP, molecular featurization, chemical ML."
agent_created: true
---

# DeepMol Skill

## Overview

DeepMol is a Python-based machine learning and deep learning framework for computational chemistry and drug discovery, built on RDKit, TensorFlow/Keras, PyTorch, scikit-learn, and DeepChem.

**Verified on:** DeepMol 1.2.1, Python 3.14, Windows 11 -- May 2026.

**Paper:** Correia, Capela & Rocha (2024). *DeepMol: An Automated Machine and Deep Learning Framework for Computational Chemistry.* Journal of Cheminformatics, 16, 136.

## When to Use This Skill

Invoke this skill when the user wants to:

- Load molecular data files (CSV with SMILES, SDF with 3D structures)
- Standardize chemical compounds (ChEMBL, custom heavy standardization)
- Generate molecular fingerprints (Morgan, MACCS, RDK, AtomPair, Layered)
- Select features or reduce dimensionality for molecular descriptors
- Train scikit-learn models for molecular property prediction
- Optimize hyperparameters with grid/random search
- Explain model predictions with SHAP
- Handle imbalanced molecular datasets
- Build end-to-end prediction pipelines
- Run automated pipeline optimization with Optuna
- Do QSAR/QSPR modeling, virtual screening, or cheminformatics ML

## Installation (Verified on Python 3.14)

### Quick Start

**Step 1:** Install DeepMol core (skip `[all]` -- it breaks on Python >= 3.13):

```bash
pip install deepmol --no-deps
```

**Step 2:** Install compatible scikit-learn first (DeepMol pins `<1.6` but 1.8 works):

```bash
pip install scikit-learn
```

**Step 3:** Install the core dependency bundle:

```bash
pip install rdkit seaborn pillow h5py imbalanced-learn chembl_structure_pipeline graph-part kneed shap umap-learn dill boruta ipython pandas biosynfoni cached_property timeout_decorator matplotlib networkx plotly transformers
```

**Step 4:** Install PyTorch (CPU version is sufficient for most tasks):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 5:** Install Optuna (optional, for automated pipeline optimization):

```bash
pip install optuna
```

**Step 6:** Verify installation:

```bash
python scripts/check_install.py
```

### What You Get vs. What's Skipped

| Component | Status | Note |
|-----------|--------|------|
| scikit-learn models | Working | RandomForest, SVM, etc. -- main backend |
| Morgan/MACCS/RDK fingerprints | Working | Core featurization |
| SHAP explainability | Working | Beeswarm, waterfall, feature plots |
| Pipeline (save/load) | Working | Without StandardScaler in pipeline |
| Optuna optimization | Working | Automated pipeline tuning |
| UMAP/PCA/KMeans | Working | Unsupervised analysis |
| SMOTE/SMOTEENN | Working | Imbalanced data handling |
| Keras/TensorFlow models | Not on 3.14 | scikeras has no 3.14 wheel; use 3.12 or conda |
| DeepChem GNN models | Not on 3.14 | Needs TensorFlow -> scikeras chain |
| Mol2Vec featurization | Needs C++ tools | gensim requires MSVC compiler |

### Critical Installation Notes

- **DO NOT install JAX** -- causes dependency conflicts.
- **DO NOT use `pip install "deepmol[all]"` on Python >= 3.13** -- scikeras dependency resolution fails.
- **Pre-trained ADMET models:** `pip install deepmol-models` (requires scikeras -> needs Python <= 3.12).
- **GPU:** Install matching CUDA + cuDNN before TensorFlow/PyTorch GPU versions.
- **Docker** alternative: `docker pull biosystemsum/deepmol` (fully configured).

## Quick Start for Beginners

After installation, the fastest way to learn is the bundled tutorial:

```bash
python scripts/tutorial.py
```

This walks through loading data, standardizing, featurizing, training, and evaluating -- with explanations at each step. Uses built-in demo data.

## Workflow: End-to-End Molecular ML

### Step 1: Data Loading

```python
from deepmol.loaders.loaders import CSVLoader
from deepmol.loaders import SDFLoader

# CSV with SMILES column
loader = CSVLoader(
    dataset_path='data.csv',
    smiles_field='smiles',       # column with SMILES strings
    id_field='id',               # column with molecule IDs
    labels_fields=['activity'],  # target column(s)
    mode='auto'                  # auto-detect classification vs regression
)
dataset = loader.create_dataset()
```

**Dataset shape:** `dataset.get_shape()` returns a 3-tuple:
```python
((n_samples,), (n_samples, n_features), (n_labels,))
# Before featurization, the middle element is None.
```

### Step 2: Compound Standardization

```python
from deepmol.standardizer import BasicStandardizer, CustomStandardizer

# Heavy standardization (recommended for most QSAR tasks)
heavy = {
    'REMOVE_ISOTOPE': True,
    'NEUTRALISE_CHARGE': True,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': True,
    'ADD_HYDROGEN': True,
    'KEKULIZE': False,
    'NEUTRALISE_CHARGE_LATE': True
}
CustomStandardizer(heavy).standardize(dataset, inplace=True)
```

### Step 3: Molecular Featurization

```python
from deepmol.compound_featurization import MorganFingerprint, MACCSkeysFingerprint, RDKFingerprint

# Morgan ECFP-like (best general-purpose)
MorganFingerprint(radius=2, size=1024).featurize(dataset, inplace=True)

# MACCS keys (166-bit, interpretable)
# MACCSkeysFingerprint().featurize(dataset, inplace=True)
```

### Step 4: Feature Selection

```python
from deepmol.feature_selection import LowVarianceFS, KbestFS

LowVarianceFS(threshold=0.01).select_features(dataset, inplace=True)
KbestFS(k=128).select_features(dataset, inplace=True)
# Note: PercentileFS is not available in this version.
```

### Step 5: Data Splitting

```python
from deepmol.splitters.splitters import SingletaskStratifiedSplitter

splitter = SingletaskStratifiedSplitter()
train, valid, test = splitter.train_valid_test_split(
    dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
)
```

### Step 6: Model Training (scikit-learn)

```python
from deepmol.models.sklearn_models import SklearnModel
from sklearn.ensemble import RandomForestClassifier

model = SklearnModel(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(train)

# Save & load
model.save('my_model')
# model = SklearnModel.load('my_model')
```

### Step 7: Model Evaluation

```python
from deepmol.metrics.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

metrics = [Metric(roc_auc_score), Metric(accuracy_score), Metric(f1_score)]
scores, _ = model.evaluate(test, metrics)
# scores is a dict: {'roc_auc_score': 0.85, 'accuracy_score': 0.82, ...}
```

### Step 8: Hyperparameter Optimization

```python
from deepmol.parameter_optimization.hyperparameter_optimization import HyperparameterOptimizerValidation

params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
optimizer = HyperparameterOptimizerValidation(
    SklearnModel(RandomForestClassifier()),
    metric=Metric(accuracy_score), maximize_metric=True,
    n_iter_search=10, params_dict=params, model_type="sklearn"
)
best_model, best_params, all_results = optimizer.fit(train, valid)
```

### Step 9: SHAP Explainability

```python
from deepmol.feature_importance import ShapValues

shap = ShapValues()
shap.fit(train, model)
shap.beeswarm_plot()
shap.sample_explanation_plot(index=0, plot_type='waterfall')
```

### Step 10: End-to-End Pipeline

```python
from deepmol.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('standardizer', BasicStandardizer()),
    ('featurizer', MorganFingerprint(radius=2, size=1024)),
    ('feature_selector', KbestFS(k=128)),
    ('model', SklearnModel(RandomForestClassifier()))
], path='my_pipeline/')

pipeline.fit_transform(train)
predictions = pipeline.predict(test)
pipeline.evaluate(test, [Metric(accuracy_score)])
pipeline.save()
# pipeline = Pipeline.load('my_pipeline/')
```

**Important:** Do NOT include `sklearn.preprocessing.StandardScaler` in the pipeline steps -- it lacks the `to_pickle` method expected by DeepMol 1.2.1's Pipeline serializer. If scaling is needed, apply it before pipeline construction.

### Step 11: Optuna Pipeline Optimization

```python
from deepmol.pipeline_optimization import PipelineOptimization

def objective(trial):
    model_type = trial.suggest_categorical('model', ['RF', 'SVC'])
    if model_type == 'RF':
        n_est = trial.suggest_int('n_estimators', 10, 200, step=10)
        model = SklearnModel(RandomForestClassifier(n_estimators=n_est))
    else:
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        model = SklearnModel(SVC(kernel=kernel))
    return [('model', model)]

po = PipelineOptimization(direction='maximize', study_name='qsar_study')
po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective,
            metric=Metric(accuracy_score), n_trials=50, save_top_n=5)
```

## Handling Imbalanced Data

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTEENN, SMOTE

train_balanced = SMOTEENN().sample(train)
```

## Unsupervised Analysis

```python
from deepmol.unsupervised import UMAP, PCA, KMeans

umap = UMAP(n_components=2, n_neighbors=15)
embedding = umap.run(dataset)
umap.plot(embedding.X, path='umap_plot.png')
```

## Troubleshooting (Verified)

| Problem | Cause | Solution |
|---------|-------|----------|
| `pip install "deepmol[all]"` fails | scikeras has no Python 3.13+ wheel | Use Step-by-step install from Installation section |
| `gensim` build fails | No MSVC C++ compiler | Skip Mol2Vec; not needed for core workflows |
| `get_shape()` returns 3 values, not 2 | DeepMol 1.2.1 API change | Unpack as `(n_samples,), (n_samples, n_feat), (n_labels,)` |
| `PercentileFS` not found | Not exported in 1.2.1 | Use `KbestFS` instead |
| `StandardScaler` in Pipeline fails to save | sklearn compat: no `to_pickle` | Remove scaler from pipeline steps |
| `ImportError: No module named 'pwd'` | Windows sandbox missing USER env | Set `USER` and `USERNAME` env vars before running Python |
| TensorFlow/Keras not importing | Python 3.14 incompatibility | Use scikit-learn backend or downgrade to Python 3.12 |
| JAX conflicts | JAX installed alongside DeepMol | `pip uninstall jax jaxlib` |
| Dataset too large for memory | Loading all molecules at once | Use `shard_size` parameter in loaders |
| SMILES parsing failures | Invalid SMILES strings in data | Apply `BasicStandardizer` before featurization |

## Bundled Resources

### scripts/

- `check_install.py` -- One-click environment verification. Run after installation to confirm everything works.
- `tutorial.py` -- Guided step-by-step tutorial for beginners. Uses built-in demo data.
- `qsar_pipeline.py` -- Full CLI QSAR pipeline: CSV -> train -> evaluate -> save.
- `batch_featurization.py` -- Batch molecular featurization CLI.
- `admet_prediction.py` -- Pre-trained ADMET model prediction CLI.

### references/

- `api_reference.md` -- Comprehensive API reference for all DeepMol modules with parameter details.

## Important Rules

1. **Verify installation first** -- `python scripts/check_install.py` before any work.
2. **Standardize before featurization** -- prevents invalid molecule errors.
3. **Use `inplace=True`** -- chains transformations on the same dataset object.
4. **Stratified splitting for classification** -- `SingletaskStratifiedSplitter`.
5. **Save models** -- `.save()` / `.load()` for reproducibility.
6. **Skip JAX** -- never install JAX with DeepMol.
7. **Python >= 3.13: no `[all]` extra** -- use step-by-step install; scikeras is the blocker.
8. **No StandardScaler in Pipeline** -- apply scaling separately if needed.
9. **Prefer `SklearnModel`** -- it's the most reliable backend across Python versions.
