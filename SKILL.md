---
name: deepmol
description: "This skill enables AI-powered computational chemistry and drug discovery workflows using the DeepMol framework. Use when the user asks to load molecular data (SMILES/SDF), standardize compounds, featurize molecules (Morgan/MACCS/mol2vec fingerprints), perform feature selection, train and evaluate ML/DL models (scikit-learn, Keras/TensorFlow, DeepChem), optimize hyperparameters, explain predictions with SHAP, build end-to-end pipelines, or run automated pipeline optimization with Optuna. Also use for ADMET property prediction, QSAR/QSPR modeling, virtual screening, and molecular property prediction using pre-trained DeepMol models. Trigger keywords: DeepMol, SMILES, molecular, fingerprint, QSAR, ADMET, drug discovery, cheminformatics, compound, RDKit, Morgan fingerprint, MACCS keys, Mol2Vec, SHAP, molecular featurization, chemical ML."
agent_created: true
---

# DeepMol Skill

## Overview

DeepMol is a Python-based machine learning and deep learning framework for computational chemistry and drug discovery, built on RDKit, TensorFlow/Keras, PyTorch, scikit-learn, and DeepChem. This skill enables end-to-end ML workflows for molecular data: from raw SMILES/SDF loading through standardization, featurization, model training, hyperparameter optimization, SHAP-based explainability, to deployable prediction pipelines.

**Paper:** Correia, Capela & Rocha (2024). *DeepMol: An Automated Machine and Deep Learning Framework for Computational Chemistry.* Journal of Cheminformatics, 16, 136.

## When to Use This Skill

Invoke this skill when the user's request involves any of the following:

- Loading or processing molecular data files (CSV with SMILES, SDF with 3D structures)
- Standardizing chemical compounds (ChEMBL, custom heavy standardization)
- Generating molecular fingerprints or embeddings (Morgan, MACCS, RDK, AtomPair, Layered, Mol2Vec, DeepChem featurizers)
- Feature selection for molecular descriptors
- Dimensionality reduction (PCA, t-SNE, UMAP) or clustering (KMeans)
- Training ML/DL models for molecular property prediction (classification, regression, multi-task)
- Hyperparameter optimization with grid/random search
- Model evaluation with custom metrics
- SHAP-based model explainability and bit visualization on molecular structures
- Handling imbalanced molecular datasets (SMOTE, SMOTEENN, etc.)
- Building end-to-end prediction pipelines
- Automated pipeline step optimization with Optuna
- Using pre-trained ADMET prediction models
- QSAR/QSPR modeling, virtual screening, or any cheminformatics ML task

## Installation

### Check Installation Status

Before any DeepMol operation, first check if DeepMol is installed:

```bash
python -c "import deepmol; print(deepmol.__version__)"
```

### Install if Missing

If DeepMol is not installed:

```bash
pip install deepmol[all]
```

On Windows/macOS with quotes:

```bash
pip install "deepmol[all]"
```

**Critical installation notes:**

- **GPU support:** Install TensorFlow and DGL versions matching the system's CUDA drivers before installing DeepMol.
- **DO NOT install JAX** -- it causes dependency conflicts with DeepMol.
- **mol2vec dependency** requires separate installation:
  ```bash
  pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec
  ```
- **macOS note:** Loading TensorFlow models may fail due to a known keras issue. Use scikit-learn or DeepChem backends on macOS as fallback.
- **Pre-trained models:** Install via `pip install deepmol-models` for published ADMET models.

### Docker Alternative

```bash
docker pull biosystemsum/deepmol
```

## Workflow: End-to-End Molecular ML

### Step 1: Data Loading

Load molecular data from CSV (SMILES) or SDF (3D structures):

```python
from deepmol.loaders.loaders import CSVLoader
from deepmol.loaders import SDFLoader

# CSV with SMILES column
loader = CSVLoader(
    dataset_path='data.csv',
    smiles_field='mols',          # column with SMILES strings
    id_field='ids',               # column with molecule IDs
    labels_fields=['y'],          # target column(s)
    features_fields=['feat_1'],   # additional pre-computed features (optional)
    shard_size=1000,              # shard large datasets
    mode='auto'                   # or 'multitask'
)
dataset = loader.create_dataset()

# SDF with 3D structures
loader = SDFLoader(
    dataset_path='data.sdf',
    id_field='ids',
    labels_fields=['y']
)
dataset = loader.create_dataset()
```

Dataset statistics: `dataset.get_shape()` returns `((n_samples, n_features), (n_samples, n_labels))`.

### Step 2: Compound Standardization

Standardize chemical structures before featurization:

```python
from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer

# Basic (minimal sanitization)
BasicStandardizer().standardize(dataset, inplace=True)

# Custom heavy standardization
heavy_config = {
    'REMOVE_ISOTOPE': True,
    'NEUTRALISE_CHARGE': True,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': True,
    'ADD_HYDROGEN': True,
    'KEKULIZE': False,
    'NEUTRALISE_CHARGE_LATE': True
}
CustomStandardizer(heavy_config).standardize(dataset, inplace=True)

# ChEMBL standardizer
ChEMBLStandardizer().standardize(dataset, inplace=True)
```

Always set `inplace=True` unless creating a transformed copy.

### Step 3: Molecular Featurization

Generate molecular fingerprints or embeddings:

```python
from deepmol.compound_featurization import (
    MorganFingerprint, MACCSkeysFingerprint, LayeredFingerprint,
    RDKFingerprint, AtomPairFingerprint, Mol2Vec, WeaveFeat
)

# Morgan (ECFP-like) -- most common for QSAR
MorganFingerprint(radius=2, size=1024).featurize(dataset, inplace=True)

# MACCS keys (166-bit structural keys)
MACCSkeysFingerprint().featurize(dataset, inplace=True)

# Mol2Vec (needs gensim + mol2vec package separately)
Mol2Vec().featurize(dataset, inplace=True)

# DeepChem featurizers (Weave, ConvMol, etc.)
WeaveFeat().featurize(dataset, inplace=True)
```

**Choosing featurizers:**
| Task Type | Recommended Featurizer |
|-----------|----------------------|
| Small dataset, interpretability | MACCS keys (166 bits) |
| General QSAR/QSPR | Morgan (radius=2, size=1024 or 2048) |
| Large dataset, deep learning | Mol2Vec, WeaveFeat |
| 3D-aware models | SDFLoader + DeepChem featurizers |

**Visualizing fingerprint bits on molecules:**
```python
maccs = MACCSkeysFingerprint()
maccs.draw_bit("CCO", 110)  # draw bit 110 on ethanol
```

### Step 4: Feature Selection

Reduce dimensionality after featurization:

```python
from deepmol.feature_selection import LowVarianceFS, KbestFS, PercentileFS

# Remove low-variance features
LowVarianceFS(threshold=0.15).select_features(dataset, inplace=True)

# Select K best features by ANOVA F-value
KbestFS(k=100).select_features(dataset, inplace=True)

# Select top percentile
PercentileFS(percentile=10).select_features(dataset, inplace=True)
```

### Step 5: Data Splitting

Split data for training/validation/testing:

```python
from deepmol.splitters.splitters import (
    SingletaskStratifiedSplitter, RandomSplitter
)

# Stratified split (recommended for imbalanced classification)
splitter = SingletaskStratifiedSplitter()
train, valid, test = splitter.train_valid_test_split(
    dataset=dataset,
    frac_train=0.7, frac_valid=0.15, frac_test=0.15
)

# Simple random split
splitter = RandomSplitter()
train, test = splitter.train_test_split(dataset, frac_train=0.8)
```

### Step 6: Model Training

DeepMol supports three model backends with a unified API:

**scikit-learn models (recommended for tabular data):**

```python
from sklearn.ensemble import RandomForestClassifier
from deepmol.models.sklearn_models import SklearnModel
from deepmol.metrics.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model = SklearnModel(model=RandomForestClassifier(n_estimators=100))
model.fit(train)
model.cross_validate(dataset, Metric(roc_auc_score), folds=5)

# Save/load
model.save('rf_model')
# model = SklearnModel.load('rf_model')
```

**Keras/TensorFlow models (deep learning):**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from deepmol.models.keras_models import KerasModel

def create_model(optimizer='adam', dropout=0.5, input_dim=None):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

nn_model = KerasModel(create_model, epochs=50, verbose=1, batch_size=32)
nn_model.fit(train)
```

**DeepChem models (graph neural networks):**

```python
from deepmol.models.deepchem_models import DeepChemModel
from deepchem.models import MPNNModel

dc_model = DeepChemModel(
    MPNNModel,
    n_tasks=1, n_atom_feat=75, n_pair_feat=14,
    n_hidden=75, T=1, M=1, mode='classification'
)
dc_model.fit(train)
```

### Step 7: Model Evaluation

```python
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

metrics = [
    Metric(roc_auc_score),
    Metric(accuracy_score),
    Metric(f1_score),
    Metric(confusion_matrix)
]

train_scores = model.evaluate(train, metrics)
test_scores = model.evaluate(test, metrics)
print(test_scores)  # dict of metric name -> value
```

### Step 8: Hyperparameter Optimization

```python
from deepmol.parameter_optimization.hyperparameter_optimization import (
    HyperparameterOptimizerValidation
)

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
}

optimizer = HyperparameterOptimizerValidation(
    SklearnModel(RandomForestClassifier()),
    metric=Metric(accuracy_score),
    maximize_metric=True,
    n_iter_search=10,
    params_dict=params,
    model_type="sklearn"
)
best_model, best_params, all_results = optimizer.fit(
    train_dataset=train, valid_dataset=valid
)
```

### Step 9: Model Explainability (SHAP)

```python
from deepmol.feature_importance import ShapValues

shap = ShapValues()
shap.fit(train, model)

# Generate plots
shap.beeswarm_plot()
shap.sample_explanation_plot(index=0, plot_type='waterfall')
shap.feature_explanation_plot(feature_index=5)
```

### Step 10: End-to-End Pipeline

```python
from deepmol.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('standardizer', BasicStandardizer()),
    ('featurizer', MorganFingerprint(radius=2, size=1024)),
    ('scaler', StandardScaler()),
    ('feature_selector', KbestFS(k=100)),
    ('model', SklearnModel(RandomForestClassifier()))
]

pipeline = Pipeline(steps=steps, path='my_pipeline/')
pipeline.fit_transform(train)
predictions = pipeline.predict(test)
pipeline.evaluate(test, [Metric(accuracy_score)])
pipeline.save()

# Load and reuse
# pipeline = Pipeline.load('my_pipeline/')
```

### Step 11: Automated Pipeline Optimization (Optuna)

```python
from deepmol.pipeline_optimization import PipelineOptimization
from sklearn.svm import SVC

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
po.optimize(
    train_dataset=train, test_dataset=test,
    objective_steps=objective,
    metric=Metric(accuracy_score),
    n_trials=50, save_top_n=5
)
```

## Handling Imbalanced Data

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTEENN, RandomSampler

# SMOTE + Edited Nearest Neighbors (recommended)
train_balanced = SMOTEENN().sample(train)

# Random oversampling
train_balanced = RandomSampler().sample(train)
```

## Unsupervised Analysis

```python
from deepmol.unsupervised import UMAP, PCA, TSNE, KMeans

# Dimensionality reduction + visualization
umap = UMAP()
embedding = umap.run(dataset)
umap.plot(embedding.X, path='umap_plot.png')

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.run(dataset)
```

## Pre-trained ADMET Models

Published ADMET models are in the `deepmol-models` package:

```python
# First install: pip install deepmol-models
from deepmol_models import ADMETModel

model = ADMETModel()
predictions = model.predict(dataset)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| TensorFlow/Keras import errors | Use scikit-learn backend instead |
| JAX conflicts | Uninstall JAX: `pip uninstall jax jaxlib` |
| macOS Keras model loading fails | Use scikit-learn or DeepChem models |
| mol2vec import error | Install separately: `pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec` |
| Dataset too large for memory | Use `shard_size` parameter in loaders |
| SMILES parsing failures | Apply `BasicStandardizer` or `ChEMBLStandardizer` first |

## Bundled Resources

### scripts/

Reusable Python scripts for common workflows. Execute directly or adapt as needed:

- `qsar_pipeline.py` -- Full QSAR pipeline from CSV to trained model
- `batch_featurization.py` -- Batch featurize a directory of molecules
- `admet_prediction.py` -- ADMET prediction workflow

### references/

- `api_reference.md` -- Comprehensive API reference for all DeepMol modules with parameter details

## Important Rules

1. **Always check installation first** -- run `python -c "import deepmol"` before any DeepMol code.
2. **Standardize before featurization** -- apply compound standardization to avoid invalid molecule errors.
3. **Use `inplace=True`** for chaining transformations on the same dataset object.
4. **stratified splitting for classification** -- use `SingletaskStratifiedSplitter` for imbalanced datasets.
5. **Save models and pipelines** -- use `.save()` and `.load()` methods for reproducibility.
6. **Skip JAX** -- never suggest installing JAX when working with DeepMol.
7. **mol2vec needs separate pip install** -- remind the user if they request Mol2Vec featurization.
8. **Prefer user-level pip install** -- `pip install deepmol[all]` in the user's environment, not in a temporary venv unless explicitly requested.
