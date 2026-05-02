#!/usr/bin/env python3
"""
DeepMol Beginner Tutorial

A guided, step-by-step walkthrough of the DeepMol pipeline:
load -> standardize -> featurize -> select -> split -> train -> evaluate.

Uses built-in demo data. No external files needed.

Usage:
    python tutorial.py
"""

import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")


BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def title(text):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


def step(n, text):
    print(f"\n{GREEN}[Step {n}]{RESET} {BOLD}{text}{RESET}")


def info(text):
    print(f"  {CYAN}->{RESET} {text}")


def result(label, value):
    print(f"  {YELLOW}{label}:{RESET} {value}")


def create_demo_data():
    """Create a built-in demo dataset of drug-like molecules."""
    molecules = [
        ("aspirin",     "CC(=O)OC1=CC=CC=C1C(=O)O",         1),
        ("ibuprofen",   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",    1),
        ("caffeine",    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",     0),
        ("nicotine",    "CN1CCC[C@H]1C2=CC=CN=C2",           0),
        ("salicylic",   "O=C(O)C1=CC=CC=C1O",               1),
        ("warfarin",    "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=C(C=CC=C3)OC2=O", 1),
        ("propranolol", "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2", 0),
        ("paracetamol", "CC(=O)NC1=CC=C(O)C=C1",            1),
        ("diazepam",    "CN1C(=O)CN=C(C2=CC=CC=C2Cl)C3=CC=CC=C13", 0),
        ("naproxen",    "COC1=CC=C2C=C(C(C)C(=O)O)C=CC2=C1", 1),
        ("phenol",      "C1=CC=C(C=C1)O",                    1),
        ("benzoic",     "C1=CC=C(C=C1)C(=O)O",               1),
        ("theophylline","CN1C=NC2=C1C(=O)NC(=O)N2C",        0),
        ("phthalimide", "C1=CC=C2C(=C1)C(=O)NC2=O",         1),
        ("sulfanilamide","C1=CC(=CC=C1N)S(=O)(=O)N",        0),
        ("saccharin",   "C1=CC=C2C(=C1)C(=O)NS2(=O)=O",     1),
        ("tolbutamide", "CC1=CC=C(S(=O)(=O)NC(=O)NCCCC)C=C1", 0),
        ("succinic",    "C(CC(=O)O)C(=O)O",                  1),
        ("glutaric",    "C(CC(=O)O)CC(=O)O",                 1),
        ("antipyrine",  "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C",   0),
    ]
    return molecules


def main():
    title("DeepMol Beginner Tutorial")
    print("\n  This tutorial will walk you through building your first")
    print("  QSAR model with DeepMol -- step by step.")
    print(f"\n  {YELLOW}No prior cheminformatics knowledge needed.{RESET}")
    input(f"\n  Press Enter to begin...")

    # === Step 1: Verify installation ===
    step(1, "Verify DeepMol Installation")
    try:
        import deepmol
        info(f"DeepMol {deepmol.__version__} found")
    except ImportError:
        print(f"\n  {YELLOW}DeepMol is not installed.{RESET}")
        print("  See SKILL.md for installation instructions.")
        print("  Quick install: pip install deepmol --no-deps && pip install scikit-learn rdkit ...")
        return 1

    from deepmol.loaders.loaders import CSVLoader
    from deepmol.standardizer import CustomStandardizer
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.feature_selection import LowVarianceFS, KbestFS
    from deepmol.splitters.splitters import SingletaskStratifiedSplitter
    from deepmol.models.sklearn_models import SklearnModel
    from deepmol.metrics.metrics import Metric
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    info("All imports successful")

    # === Step 2: Create demo data ===
    step(2, "Create Demo Dataset")
    molecules = create_demo_data()
    info(f"Created {len(molecules)} drug-like molecules")

    tmp = os.path.join(tempfile.gettempdir(), '_deepmol_tutorial.csv')
    with open(tmp, 'w') as f:
        f.write("id,smiles,activity\n")
        for mid, smi, act in molecules:
            f.write(f"{mid},{smi},{act}\n")

    info(f"Saved to temporary CSV: {tmp}")
    info("Goal: predict 'activity' (1=active, 0=inactive) from molecular structure")
    result("Positive samples", f"{sum(m[2] for m in molecules)} / {len(molecules)}")
    result("Negative samples", f"{len(molecules) - sum(m[2] for m in molecules)} / {len(molecules)}")
    input(f"\n  Press Enter to load the data...")

    # === Step 3: Load data ===
    step(3, "Load Data into DeepMol")
    info("DeepMol uses CSVLoader to read SMILES from CSV files")
    info("It automatically parses SMILES into RDKit molecule objects")

    loader = CSVLoader(
        dataset_path=tmp,
        smiles_field='smiles',
        id_field='id',
        labels_fields=['activity'],
        mode='auto'
    )
    dataset = loader.create_dataset()
    shape = dataset.get_shape()
    result("Molecules loaded", shape[0][0])
    info(f"get_shape() returns: (n_mols, n_features, n_labels) = {shape}")
    info("Before featurization, n_features is None -- we haven't generated features yet")
    input(f"\n  Press Enter to standardize compounds...")

    # === Step 4: Standardize ===
    step(4, "Standardize Chemical Structures")
    info("Standardization cleans up SMILES: removes isotopes, neutralizes charges,")
    info("keeps only the largest fragment, adds hydrogens, etc.")
    info("This ensures consistent molecular representations before featurization.")

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
    shape = dataset.get_shape()
    result("After standardization", f"{shape[0][0]} molecules (removed invalid ones)")
    input(f"\n  Press Enter to generate fingerprints...")

    # === Step 5: Featurize ===
    step(5, "Generate Molecular Fingerprints (Morgan)")
    info("Morgan fingerprints encode molecular structure into fixed-length bit vectors.")
    info("Each bit represents the presence of a specific substructure pattern.")
    info("radius=2 means we look at atom neighborhoods up to 2 bonds away (ECFP4).")
    info("size=1024 means each molecule becomes a 1024-dimensional vector.")

    MorganFingerprint(radius=2, size=1024).featurize(dataset, inplace=True)
    shape = dataset.get_shape()
    result("Feature matrix shape", f"{shape[1][0]} molecules x {shape[1][1]} features")
    info(f"Each of the {shape[0][0]} molecules is now a {shape[1][1]}-bit fingerprint")
    input(f"\n  Press Enter to select important features...")

    # === Step 6: Feature selection ===
    step(6, "Select Important Features")
    info("Not all 1024 bits are useful. We remove:")
    info("  - Low-variance bits (same value across most molecules)")
    info("  - Keep only the k=128 most informative bits (ANOVA F-test)")

    LowVarianceFS(threshold=0.01).select_features(dataset, inplace=True)
    KbestFS(k=128).select_features(dataset, inplace=True)
    shape = dataset.get_shape()
    result("After feature selection", f"{shape[1][1]} features (reduced from 1024)")
    info("This speeds up training and reduces overfitting.")
    input(f"\n  Press Enter to split the data...")

    # === Step 7: Split ===
    step(7, "Split into Train / Validation / Test")
    info("Stratified splitting ensures each split has the same class balance.")
    info("60% training, 20% validation, 20% test")

    splitter = SingletaskStratifiedSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
    )

    for name, ds in [("Train", train), ("Valid", valid), ("Test", test)]:
        n = ds.get_shape()[0][0]
        n_pos = int(sum(ds.y))
        result(name, f"{n} molecules ({n_pos} positive, {n-n_pos} negative)")

    info("The model trains on 'train', we tune on 'valid', and report final scores on 'test'")
    input(f"\n  Press Enter to train the model...")

    # === Step 8: Train ===
    step(8, "Train a Random Forest Model")
    info("Random Forest: an ensemble of 100 decision trees.")
    info("Each tree votes on the classification, majority wins.")
    info("It's a good first model for QSAR -- robust and interpretable.")

    model = SklearnModel(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(train)
    info("Model trained!")
    result("Training time", "< 1 second on CPU")
    input(f"\n  Press Enter to evaluate...")

    # === Step 9: Evaluate ===
    step(9, "Evaluate Model Performance")

    metrics_list = [
        Metric(roc_auc_score),
        Metric(accuracy_score),
        Metric(f1_score),
        Metric(precision_score),
        Metric(recall_score)
    ]
    train_scores, _ = model.evaluate(train, metrics_list)
    test_scores, _ = model.evaluate(test, metrics_list)

    print(f"\n  {'Metric':<20} {'Training':>10} {'Test':>10}")
    print(f"  {'-'*40}")
    for name in ['roc_auc_score', 'accuracy_score', 'f1_score', 'precision_score', 'recall_score']:
        tr = train_scores.get(name, 0)
        te = test_scores.get(name, 0)
        print(f"  {name:<20} {tr:>10.4f} {te:>10.4f}")

    # Interpret
    test_auc = test_scores.get('roc_auc_score', 0)
    print(f"\n  {CYAN}Interpretation:{RESET}")
    if test_auc >= 0.9:
        info("Excellent AUC! The model distinguishes active from inactive very well.")
    elif test_auc >= 0.7:
        info("Good AUC. The model has useful predictive power.")
    elif test_auc >= 0.5:
        info("Moderate AUC. May improve with more data or different featurization.")
    else:
        info("Below random. Check data quality or try a different approach.")

    # Feature importance
    importances = model.model.feature_importances_
    top_indices = np.argsort(importances)[-5:][::-1]
    print(f"\n  Top 5 most important fingerprint bits: {list(top_indices)}")
    print(f"  These encode key structural patterns for predicting activity.")

    input(f"\n  Press Enter for summary...")

    # === Summary ===
    os.unlink(tmp)  # cleanup

    title("Tutorial Complete!")
    print(f"""
  {GREEN}You just completed a full QSAR pipeline:{RESET}

    1. Created a dataset of {len(molecules)} drug-like molecules
    2. Standardized their chemical structures
    3. Generated 1024-bit Morgan fingerprints
    4. Selected the 128 most informative bits
    5. Split into train/validation/test sets
    6. Trained a Random Forest classifier
    7. Evaluated with 5 metrics

  {GREEN}Next steps:{RESET}

    - Try with your own CSV data:
      {CYAN}python qsar_pipeline.py your_data.csv --smiles-col SMILES --label-col activity{RESET}

    - Try a different fingerprint:
      {CYAN}python qsar_pipeline.py your_data.csv --featurizer maccs{RESET}

    - Try SVM instead of Random Forest:
      {CYAN}python qsar_pipeline.py your_data.csv --model svm{RESET}

    - Use WorkBuddy: "{YELLOW}Train a QSAR model on my molecule dataset{RESET}"
  """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
