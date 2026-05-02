# DeepMol API Reference

Comprehensive module-by-module API reference with all parameters and return types.

---

## `deepmol.loaders` -- Data Loading

### CSVLoader

```python
from deepmol.loaders.loaders import CSVLoader

CSVLoader(
    dataset_path: str,          # path to CSV file
    smiles_field: str,          # column name for SMILES strings
    id_field: str,              # column name for molecule IDs
    labels_fields: List[str],   # list of target column names
    features_fields: List[str], # list of additional feature columns (optional)
    shard_size: int = 1000,     # rows per shard for large datasets
    mode: str = 'auto'          # 'auto' for single-task, 'multitask' for multi-task
)
# Returns: Loader instance
# .create_dataset() -> Dataset  (with .mols, .ids, .y, .X attributes)
```

### SDFLoader

```python
from deepmol.loaders import SDFLoader

SDFLoader(
    dataset_path: str,          # path to SDF file
    id_field: str,              # SDF property name for IDs
    labels_fields: List[str],   # SDF property names for labels
    features_fields: List[str], # SDF property names for features
    shard_size: int = 1000,
    mode: str = 'auto'
)
# .create_dataset() -> Dataset
```

### Loaders (generic)

```python
from deepmol.loaders.loaders import Loaders

Loaders.load(
    path: str,                  # file path
    format: str = 'csv',        # 'csv', 'sdf', 'tsv'
    **kwargs                    # passed to specific loader
)
```

---

## `deepmol.standardizer` -- Compound Standardization

### BasicStandardizer

```python
from deepmol.standardizer import BasicStandardizer

BasicStandardizer()
# .standardize(dataset: Dataset, inplace: bool = False) -> Optional[Dataset]
#  - Performs basic RDKit sanitization only (valence checks, kekulization, aromaticity)
```

### CustomStandardizer

```python
from deepmol.standardizer import CustomStandardizer

CustomStandardizer(
    heavy_standardization: dict = {
        'REMOVE_ISOTOPE': True,           # strip isotope info
        'NEUTRALISE_CHARGE': True,        # neutralize formal charges
        'REMOVE_STEREO': True,            # remove stereochemistry
        'KEEP_BIGGEST': True,            # keep only largest fragment
        'ADD_HYDROGEN': True,            # add explicit hydrogens
        'KEKULIZE': False,               # convert aromatic bonds to kekule
        'NEUTRALISE_CHARGE_LATE': True    # re-neutralize after all steps
    }
)
# .standardize(dataset, inplace=False) -> Optional[Dataset]
```

### ChEMBLStandardizer

```python
from deepmol.standardizer import ChEMBLStandardizer

ChEMBLStandardizer()
# .standardize(dataset, inplace=False) -> Optional[Dataset]
#  - Uses ChEMBL structure curation pipeline (molvs)
```

---

## `deepmol.compound_featurization` -- Molecular Featurization

### MorganFingerprint

```python
from deepmol.compound_featurization import MorganFingerprint

MorganFingerprint(
    radius: int = 2,           # fingerprint radius (2 = ECFP4, 3 = ECFP6)
    size: int = 2048,           # bit vector length
    chiral: bool = False,       # include chirality
    bonds: bool = True,         # use bond-based invariants
    features: bool = False,     # use pharmacophoric feature-based invariants
    useFeatures: bool = False
)
# .featurize(dataset: Dataset, inplace: bool = False) -> Optional[Dataset]
```

### MACCSkeysFingerprint

```python
from deepmol.compound_featurization import MACCSkeysFingerprint

MACCSkeysFingerprint()
# .featurize(dataset, inplace=False) -> Optional[Dataset]
#  166-bit structural keys
# .draw_bit(smiles: str, bit_number: int)  -- draw a specific MACCS bit on a molecule
```

### LayeredFingerprint

```python
from deepmol.compound_featurization import LayeredFingerprint

LayeredFingerprint(
    layerFlags=None,           # RDKit LayeredFingerprint layer flags
    minPath: int = 1,
    maxPath: int = 7,
    fpSize: int = 2048,
    atomCounts: list = None,
    branchedPaths: bool = True
)
```

### RDKFingerprint

```python
from deepmol.compound_featurization import RDKFingerprint

RDKFingerprint(
    minPath: int = 1,
    maxPath: int = 7,
    fpSize: int = 2048,
    nBitsPerHash: int = 2,
    useHs: bool = True,
    tgtDensity: float = 0.0,
    minSize: int = 128
)
```

### AtomPairFingerprint

```python
from deepmol.compound_featurization import AtomPairFingerprint

AtomPairFingerprint(
    nBits: int = 2048,
    minLength: int = 1,
    maxLength: int = 30
)
```

### Mol2Vec

```python
from deepmol.compound_featurization import Mol2Vec

Mol2Vec(
    model_path: str = None,    # path to pretrained mol2vec model file
    radius: int = 1,
    unseen: str = 'UNK'        # handling for unseen substructures
)
# Requires: pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec
```

### DeepChem Featurizers

```python
from deepmol.compound_featurization import (
    WeaveFeat, ConvMolFeat, GraphConvFeat, CoulombFeat
)

WeaveFeat(
    max_atoms: int = 75,
    max_pairs: int = 75,
    graph_distance: bool = True,
    explicit_h: bool = False
)

ConvMolFeat(
    max_atoms: int = 75,
    master_atom: bool = False
)
```

---

## `deepmol.feature_selection` -- Feature Selection

### LowVarianceFS

```python
from deepmol.feature_selection import LowVarianceFS

LowVarianceFS(
    threshold: float = 0.0    # variance threshold (0.0 = remove constant features)
)
# .select_features(dataset, inplace=False) -> Optional[Dataset]
```

### KbestFS

```python
from deepmol.feature_selection import KbestFS

KbestFS(
    k: int = 10,              # number of top features to keep
    score_func = f_classif    # scoring function (default: ANOVA F-value)
)
# .select_features(dataset, inplace=False) -> Optional[Dataset]
```

### PercentileFS

```python
from deepmol.feature_selection import PercentileFS

PercentileFS(
    percentile: int = 10,     # top percentile to keep
    score_func = f_classif
)
```

### RecursiveFeatureElimination

```python
from deepmol.feature_selection import RecursiveFeatureEliminationFS

RecursiveFeatureEliminationFS(
    model,                     # fitted sklearn-compatible model with coef_ or feature_importances_
    n_features_to_select: int = None,  # final number of features
    step: int = 1              # features to remove per iteration
)
```

### ImportanceFS

```python
from deepmol.feature_selection import ImportanceFS

ImportanceFS(
    model,                     # fitted model with feature_importances_
    threshold: float = 0.01    # minimum importance to keep
)
```

---

## `deepmol.unsupervised` -- Dimensionality Reduction & Clustering

### PCA

```python
from deepmol.unsupervised import PCA

PCA(
    n_components: int = 2,
    **kwargs                   # passed to sklearn PCA
)
# .run(dataset) -> PCA result with .X (transformed data)
# .plot(data, path=None, title=None, labels=None, cmap=None)
```

### TSNE

```python
from deepmol.unsupervised import TSNE

TSNE(
    n_components: int = 2,
    perplexity: float = 30.0,
    **kwargs
)
# .run(dataset) -> result with .X
# .plot(data, path=None, ...)
```

### UMAP

```python
from deepmol.unsupervised import UMAP

UMAP(
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    **kwargs
)
# .run(dataset) -> result with .X
# .plot(data, path=None, ...)
```

### KMeans

```python
from deepmol.unsupervised import KMeans

KMeans(
    n_clusters: int = 8,
    random_state: int = 42,
    **kwargs
)
# .run(dataset) -> result with .labels_
# .plot(data, labels, path=None)
```

---

## `deepmol.splitters` -- Data Splitting

### SingletaskStratifiedSplitter

```python
from deepmol.splitters.splitters import SingletaskStratifiedSplitter

SingletaskStratifiedSplitter()
# .train_test_split(dataset, frac_train=0.8, seed=None) -> (train, test)
# .train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.15, frac_test=0.15, seed=None) -> (train, valid, test)
# .k_fold_split(dataset, k=5, seed=None) -> List[(train, test)]
```

### RandomSplitter

```python
from deepmol.splitters.splitters import RandomSplitter

RandomSplitter()
# .train_test_split(dataset, frac_train=0.8, seed=None)
# .train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.15, frac_test=0.15, seed=None)
```

### SingletaskScaffoldSplitter

```python
from deepmol.splitters.splitters import SingletaskScaffoldSplitter

SingletaskScaffoldSplitter()
# Uses Bemis-Murcko scaffolds for splitting -- better for generalization evaluation
# .train_test_split(dataset, frac_train=0.8, seed=None)
# .train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.15, frac_test=0.15, seed=None)
```

---

## `deepmol.models` -- Model Building

### SklearnModel

```python
from deepmol.models.sklearn_models import SklearnModel

SklearnModel(
    model: Any,                  # sklearn-compatible model instance
    mode: str = 'classification', # 'classification' or 'regression'
    model_dir: str = None,       # directory for saving model artifacts
    **kwargs
)
# .fit(dataset: Dataset)
# .predict(dataset: Dataset) -> np.ndarray
# .predict_proba(dataset: Dataset) -> np.ndarray
# .evaluate(dataset: Dataset, metrics: List[Metric]) -> Dict[str, float]
# .cross_validate(dataset: Dataset, metric: Metric, folds: int = 5) -> Tuple[List[float], float, float]
# .save(path: str)
# @classmethod .load(path: str) -> SklearnModel
```

### KerasModel

```python
from deepmol.models.keras_models import KerasModel

KerasModel(
    model_builder: Callable,     # function that returns a compiled Keras model
    mode: str = 'classification', # 'classification' or 'regression'
    epochs: int = 150,
    batch_size: int = 32,
    verbose: int = 0,
    **kwargs                     # passed to model_builder
)
# The model_builder function signature:
#   def model_builder(input_dim=None, **kwargs) -> tf.keras.Model
#
# .fit(dataset, validation_dataset=None)
# .predict(dataset) -> np.ndarray
# .predict_proba(dataset) -> np.ndarray
# .evaluate(dataset, metrics) -> dict
# .save(path)
# @classmethod .load(path) -> KerasModel
```

### DeepChemModel

```python
from deepmol.models.deepchem_models import DeepChemModel

DeepChemModel(
    model_class,                 # DeepChem model class (e.g., MPNNModel, GraphConvModel)
    mode: str = 'classification',
    **kwargs                     # passed to model_class constructor
)
# .fit(dataset)
# .predict(dataset) -> np.ndarray
# .evaluate(dataset, metrics) -> dict
# .save(path)
# @classmethod .load(path) -> DeepChemModel
```

### EnsembleModel

```python
from deepmol.models.ensembles import EnsembleModel

EnsembleModel(
    models: List,                # list of DeepMol model instances
    voting: str = 'soft'         # 'soft' or 'hard'
)
# .fit(dataset)
# .predict(dataset) -> np.ndarray
```

---

## `deepmol.metrics` -- Evaluation Metrics

### Metric

```python
from deepmol.metrics.metrics import Metric

Metric(
    metric: Callable,            # sklearn-compatible metric function
    name: str = None             # display name (auto-generated if None)
)
# Common sklearn metric functions used:
#   roc_auc_score, accuracy_score, precision_score, recall_score,
#   f1_score, confusion_matrix, classification_report,
#   mean_squared_error, mean_absolute_error, r2_score
```

### Prebuilt Metric Functions

```python
from deepmol.metrics.metrics_functions import (
    roc_auc_score,              # g3c standard roc_auc_score
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score
)
```

---

## `deepmol.parameter_optimization` -- Hyperparameter Tuning

### HyperparameterOptimizerValidation

```python
from deepmol.parameter_optimization.hyperparameter_optimization import HyperparameterOptimizerValidation

HyperparameterOptimizerValidation(
    model_builder,               # model instance or builder function (for Keras)
    metric: Metric,
    maximize_metric: bool = True,
    n_iter_search: int = 10,
    params_dict: dict,           # parameter grid
    model_type: str = 'sklearn', # 'sklearn' or 'keras'
    seed: int = None
)
# .fit(train_dataset, valid_dataset=None) -> Tuple[best_model, best_params, all_results]
#
# params_dict example:
#   {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
```

### HyperparameterOptimizerCV

```python
from deepmol.parameter_optimization.hyperparameter_optimization import HyperparameterOptimizerCV

HyperparameterOptimizerCV(
    model_builder,
    metric: Metric,
    maximize_metric: bool = True,
    n_iter_search: int = 10,
    params_dict: dict,
    model_type: str = 'sklearn',
    cv: int = 5,
    seed: int = None
)
# .fit(train_dataset) -> Tuple[best_model, best_params, all_results]
```

---

## `deepmol.feature_importance` -- Explainability (SHAP)

### ShapValues

```python
from deepmol.feature_importance import ShapValues

ShapValues(
    model_type: str = 'sklearn', # 'sklearn', 'keras', or 'deepchem'
    shap_explainer: str = 'auto' # 'auto', 'tree', 'deep', 'kernel', 'linear'
)
# .fit(dataset: Dataset, model=None)
# .beeswarm_plot(max_display: int = 20, show: bool = True)
# .sample_explanation_plot(index: int, plot_type: str = 'waterfall', max_display: int = 20, show: bool = True)
#   plot_type: 'waterfall', 'force', 'decision'
# .feature_explanation_plot(feature_index: int, show: bool = True)
# .summary_plot(max_display: int = 20, show: bool = True)
# .get_shap_values() -> np.ndarray
# .get_feature_names() -> List[str]

# For drawing fingerprint bits on molecular structures:
from deepmol.compound_featurization import MACCSkeysFingerprint
maccs = MACCSkeysFingerprint()
maccs.draw_bit(smiles="CCO", pattern_number=110)
```

---

## `deepmol.imbalanced_learn` -- Handling Imbalanced Data

### RandomSampler

```python
from deepmol.imbalanced_learn.imbalanced_learn import RandomSampler

RandomSampler(
    random_state: int = None
)
# .sample(dataset: Dataset) -> Dataset
```

### SMOTE

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTE

SMOTE(
    k_neighbors: int = 5,
    random_state: int = None
)
# .sample(dataset) -> Dataset
```

### SMOTEENN

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTEENN

SMOTEENN(
    random_state: int = None
)
# .sample(dataset) -> Dataset
# Combines SMOTE oversampling + Edited Nearest Neighbors cleaning
```

### SMOTETomek

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTETomek

SMOTETomek(
    random_state: int = None
)
# .sample(dataset) -> Dataset
# Combines SMOTE + Tomek links cleaning
```

### ClusterCentroids

```python
from deepmol.imbalanced_learn.imbalanced_learn import ClusterCentroids

ClusterCentroids(
    random_state: int = None
)
# .sample(dataset) -> Dataset
# Undersampling with cluster centroids
```

---

## `deepmol.pipeline` -- End-to-End Pipelines

### Pipeline

```python
from deepmol.pipeline import Pipeline

Pipeline(
    steps: List[Tuple[str, Any]],  # list of (name, transformer/model) tuples
    path: str = None,              # directory for saving/loading
    mode: str = 'classification'   # 'classification' or 'regression'
)
# .fit(dataset: Dataset)
# .predict(dataset: Dataset) -> np.ndarray
# .predict_proba(dataset: Dataset) -> np.ndarray
# .evaluate(dataset: Dataset, metrics: List[Metric]) -> Dict
# .save(path: str = None)
# @classmethod .load(path: str) -> Pipeline
```

### PipelineDataProcessing

```python
from deepmol.pipeline import PipelineDataProcessing

PipelineDataProcessing(
    steps: List[Tuple[str, Any]],
    path: str = None
)
# .fit_transform(dataset) -> Dataset
# .transform(dataset) -> Dataset
# .save(path)
# @classmethod .load(path)
```

---

## `deepmol.pipeline_optimization` -- Automated Pipeline Tuning

### PipelineOptimization

```python
from deepmol.pipeline_optimization import PipelineOptimization

PipelineOptimization(
    direction: str = 'maximize',  # 'maximize' or 'minimize'
    study_name: str = 'deepmol_study',
    storage: str = None,         # Optuna storage URL (None = in-memory)
    load_if_exists: bool = False
)
# .optimize(
#     train_dataset: Dataset,
#     test_dataset: Dataset,
#     objective_steps: Callable,  # function(trial) -> List[(name, step)]
#     metric: Metric,
#     n_trials: int = 100,
#     save_top_n: int = 5,
#     path: str = None,
#     data_processing_steps: List = None
# ) -> Tuple[best_pipeline, best_score, study]
#
# The objective_steps callable receives an Optuna Trial object
# and must return a list of (step_name, step_instance) tuples.
```

---

## `deepmol.datasets` -- Dataset Class

### Dataset

The core data container passed between all pipeline steps.

```python
from deepmol.datasets import Dataset

# Attributes:
dataset.mols      # List[rdkit.Chem.Mol] -- RDKit molecule objects
dataset.ids       # List[str] -- molecule identifiers
dataset.y         # np.ndarray -- labels/target values
dataset.X         # np.ndarray -- feature matrix (None before featurization)
dataset.features  # np.ndarray -- additional pre-computed features

# Methods:
dataset.get_shape()     # -> ((n_samples, n_features), (n_samples, n_labels))
dataset.get_n_samples() # -> int
dataset.get_n_features() # -> int
dataset.get_n_labels()   # -> int
dataset.select(ids: List[str]) -> Dataset  # subset by molecule IDs
dataset.remove(ids: List[str]) -> Dataset  # remove molecules by ID
```

---

## `deepmol.pre_trained_models` -- Published ADMET Models

```python
# Install: pip install deepmol-models

from deepmol_models import (
    BBBPModel,      # Blood-Brain Barrier Penetration
    CYP2C9Model,    # CYP2C9 Inhibition
    CYP2D6Model,    # CYP2D6 Inhibition
    CYP3A4Model,    # CYP3A4 Inhibition
    HIA_Model,      # Human Intestinal Absorption
    PgpModel,       # P-glycoprotein Substrate
    SolubilityModel # Aqueous Solubility
)

model = BBBPModel()
predictions = model.predict(dataset)
```
