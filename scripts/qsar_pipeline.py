#!/usr/bin/env python3
"""
Full QSAR/QSPR Pipeline Script

Loads a CSV with SMILES column, standardizes compounds, generates fingerprints,
trains a model, evaluates it, and saves the pipeline for reuse.

Usage:
    python qsar_pipeline.py data.csv --smiles-col SMILES --label-col activity \\
        --featurizer morgan --model rf --output my_pipeline/

Arguments:
    data.csv              : Input CSV file with SMILES strings
    --smiles-col SMILES   : Column name for SMILES (default: 'SMILES')
    --label-col activity  : Column name for target label (default: 'label')
    --id-col id           : Column name for IDs (default: 'id')
    --featurizer morgan   : Featurizer: morgan, maccs, rdk, atompair, layered (default: morgan)
    --model rf            : Model: rf (RandomForest), svm, xgb, nn (default: rf)
    --task classification : Task type: classification, regression (default: classification)
    --fp-size 1024        : Fingerprint size (default: 2048)
    --output pipeline/    : Output directory for saved pipeline (default: qsar_pipeline/)
    --no-standardize      : Skip compound standardization
    --no-feature-select   : Skip feature selection
    --no-save             : Do not save the pipeline
    --shap                : Generate SHAP explanation plots
"""

import argparse
import os
import sys

import numpy as np


def check_deepmol():
    """Verify DeepMol is installed."""
    try:
        import deepmol
        print(f"[OK] DeepMol {deepmol.__version__} is installed")
        return True
    except ImportError:
        print("[ERROR] DeepMol is not installed.")
        print("  Install: pip install deepmol[all]")
        return False


def build_pipeline(args):
    """Build the full QSAR pipeline based on CLI arguments."""
    from deepmol.loaders.loaders import CSVLoader
    from deepmol.standardizer import BasicStandardizer, CustomStandardizer
    from deepmol.compound_featurization import (
        MorganFingerprint, MACCSkeysFingerprint, RDKFingerprint,
        AtomPairFingerprint, LayeredFingerprint
    )
    from deepmol.feature_selection import LowVarianceFS, KbestFS
    from deepmol.models.sklearn_models import SklearnModel
    from deepmol.models.keras_models import KerasModel
    from deepmol.splitters.splitters import SingletaskStratifiedSplitter, RandomSplitter
    from deepmol.metrics.metrics import Metric
    from deepmol.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    # --- 1. Load data ---
    print(f"\n[1/8] Loading data from {args.data}...")
    loader = CSVLoader(
        dataset_path=args.data,
        smiles_field=args.smiles_col,
        id_field=args.id_col,
        labels_fields=[args.label_col],
        mode='auto'
    )
    dataset = loader.create_dataset()
    n_samples, (_, n_labels) = dataset.get_shape()
    print(f"  Loaded {n_samples} molecules, {n_labels} label(s)")

    # --- 2. Standardize ---
    if not args.no_standardize:
        print("[2/8] Standardizing compounds...")
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
        n_after, _ = dataset.get_shape()
        print(f"  {n_samples - n_after[0]} molecules removed during standardization")
    else:
        print("[2/8] Skipping standardization (--no-standardize)")

    # --- 3. Featurize ---
    print(f"[3/8] Generating fingerprints (featurizer={args.featurizer})...")
    featurizer_map = {
        'morgan': MorganFingerprint(radius=2, size=args.fp_size),
        'maccs': MACCSkeysFingerprint(),
        'rdk': RDKFingerprint(fpSize=args.fp_size),
        'atompair': AtomPairFingerprint(nBits=args.fp_size),
        'layered': LayeredFingerprint(fpSize=args.fp_size),
    }
    featurizer = featurizer_map[args.featurizer]
    featurizer.featurize(dataset, inplace=True)
    _, (n_features, _) = dataset.get_shape()
    print(f"  Generated {n_features} features")

    # --- 4. Scale ---
    print("[4/8] Scaling features...")
    scaler = StandardScaler()

    # --- 5. Feature selection ---
    if not args.no_feature_select and args.featurizer != 'maccs':
        print("[5/8] Selecting features (low variance + k-best)...")
        low_var = LowVarianceFS(threshold=0.01)
        kbest = KbestFS(k=min(512, n_features))
        steps = [('standardizer', BasicStandardizer()),
                 ('featurizer', featurizer_map[args.featurizer]),
                 ('scaler', scaler),
                 ('low_var', low_var),
                 ('kbest', kbest)]
    else:
        print("[5/8] Skipping feature selection")
        steps = [('standardizer', BasicStandardizer()),
                 ('featurizer', featurizer_map[args.featurizer]),
                 ('scaler', scaler)]

    # --- 6. Split ---
    print("[6/8] Splitting data...")
    if args.task == 'classification':
        splitter = SingletaskStratifiedSplitter()
    else:
        splitter = RandomSplitter()
    train, test = splitter.train_test_split(dataset, frac_train=0.8)
    print(f"  Train: {train.get_n_samples()}, Test: {test.get_n_samples()}")

    # --- 7. Model ---
    print(f"[7/8] Training model (type={args.model}, task={args.task})...")
    if args.model == 'rf':
        if args.task == 'regression':
            model = SklearnModel(RandomForestRegressor(n_estimators=100, random_state=42),
                                 mode='regression')
        else:
            model = SklearnModel(RandomForestClassifier(n_estimators=100, random_state=42))
    elif args.model == 'svm':
        if args.task == 'regression':
            model = SklearnModel(SVR(), mode='regression')
        else:
            model = SklearnModel(SVC(probability=True, random_state=42))
    elif args.model == 'nn':
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        def create_model(input_dim=None, **kwargs):
            m = Sequential([
                Dense(64, input_dim=input_dim, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid' if args.task == 'classification' else 'linear')
            ])
            loss = 'binary_crossentropy' if args.task == 'classification' else 'mse'
            m.compile(loss=loss, optimizer='adam', metrics=['accuracy'] if args.task == 'classification' else ['mae'])
            return m

        model = KerasModel(create_model, epochs=30, verbose=1, batch_size=32,
                           mode=args.task)
    else:
        print(f"[ERROR] Unknown model type: {args.model}")
        sys.exit(1)

    steps.append(('model', model))

    # --- 8. Pipeline ---
    print("[8/8] Building and evaluating pipeline...")
    pipeline = Pipeline(steps=steps, path=args.output)
    pipeline.fit_transform(train)

    # Evaluation
    if args.task == 'classification':
        metrics = [Metric(roc_auc_score), Metric(accuracy_score),
                   Metric(f1_score), Metric(precision_score), Metric(recall_score)]
    else:
        metrics = [Metric(r2_score), Metric(mean_squared_error),
                   Metric(mean_absolute_error)]

    train_results = pipeline.evaluate(train, metrics)
    test_results = pipeline.evaluate(test, metrics)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Training set ({train.get_n_samples()} samples):")
    for name, val in train_results.items():
        print(f"    {name}: {val:.4f}" if isinstance(val, float) else f"    {name}: {val}")
    print(f"\n  Test set ({test.get_n_samples()} samples):")
    for name, val in test_results.items():
        print(f"    {name}: {val:.4f}" if isinstance(val, float) else f"    {name}: {val}")

    # --- Save ---
    if not args.no_save:
        pipeline.save()
        print(f"\n[SAVED] Pipeline saved to {os.path.abspath(args.output)}/")

    # --- SHAP ---
    if args.shap and args.model in ('rf', 'svm'):
        print("\n[SHAP] Generating explainability plots...")
        try:
            from deepmol.feature_importance import ShapValues
            shap = ShapValues()
            shap.fit(train, pipeline.steps[-1][1])
            import matplotlib
            matplotlib.use('Agg')
            shap.beeswarm_plot(show=False)
            shap_path = os.path.join(args.output, 'shap_beeswarm.png')
            import matplotlib.pyplot as plt
            plt.savefig(shap_path, dpi=150, bbox_inches='tight')
            print(f"  SHAP beeswarm plot saved to {shap_path}")
        except Exception as e:
            print(f"  [WARNING] SHAP failed: {e}")

    print("\nDone.")
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description='QSAR/QSPR Pipeline with DeepMol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qsar_pipeline.py data.csv --smiles-col SMILES --label-col pIC50 --task regression
  python qsar_pipeline.py data.csv --featurizer maccs --model svm --shap
  python qsar_pipeline.py data.sdf --model nn --fp-size 512
        """
    )
    parser.add_argument('data', help='Input CSV (or SDF) file')
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES column name')
    parser.add_argument('--label-col', default='label', help='Label column name')
    parser.add_argument('--id-col', default='id', help='ID column name')
    parser.add_argument('--featurizer', default='morgan',
                        choices=['morgan', 'maccs', 'rdk', 'atompair', 'layered'])
    parser.add_argument('--model', default='rf', choices=['rf', 'svm', 'nn'])
    parser.add_argument('--task', default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--fp-size', type=int, default=2048)
    parser.add_argument('--output', default='qsar_pipeline')
    parser.add_argument('--no-standardize', action='store_true')
    parser.add_argument('--no-feature-select', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--shap', action='store_true',
                        help='Generate SHAP explainability plots')

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] File not found: {args.data}")
        sys.exit(1)

    if not check_deepmol():
        sys.exit(1)

    build_pipeline(args)


if __name__ == '__main__':
    main()
