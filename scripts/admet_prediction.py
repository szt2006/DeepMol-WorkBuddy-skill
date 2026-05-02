#!/usr/bin/env python3
"""
ADMET Property Prediction Script

Uses pre-trained DeepMol ADMET models to predict pharmacokinetic properties
for a set of compounds.

Supported models (from deepmol-models package):
  - BBBP       Blood-Brain Barrier Penetration
  - CYP2C9     CYP2C9 Inhibition
  - CYP2D6     CYP2D6 Inhibition
  - CYP3A4     CYP3A4 Inhibition
  - HIA        Human Intestinal Absorption
  - Pgp        P-glycoprotein Substrate
  - Solubility Aqueous Solubility

Usage:
    python admet_prediction.py data.csv --models BBBP CYP2D6 HIA Solubility --output predictions.csv
    python admet_prediction.py data.csv --models all --smiles-col SMILES
"""

import argparse
import os
import sys

import numpy as np

MODEL_NAMES = ['BBBP', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'HIA', 'Pgp', 'Solubility']


def check_deepmol():
    try:
        import deepmol
        return True
    except ImportError:
        print("[ERROR] DeepMol not installed.")
        print("  pip install deepmol[all]")
        return False


def check_deepmol_models():
    try:
        import deepmol_models
        print("[OK] deepmol-models is installed")
        return True
    except ImportError:
        print("[WARNING] deepmol-models not installed.")
        print("  pip install deepmol-models")
        print("  Continuing may fail...")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='ADMET Property Prediction with DeepMol pre-trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models: {', '.join(MODEL_NAMES)}

Examples:
  python admet_prediction.py library.csv --models BBBP CYP2D6 --output admet.csv
  python admet_prediction.py compounds.csv --models all
  python admet_prediction.py data.csv --models Solubility --smiles-col molecule
        """
    )
    parser.add_argument('input', help='Input CSV file with SMILES')
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES column name')
    parser.add_argument('--id-col', default=None, help='ID column name')
    parser.add_argument('--models', nargs='+', required=True,
                        help=f'Models to run, or "all". Options: {MODEL_NAMES}')
    parser.add_argument('--output', default='admet_predictions.csv',
                        help='Output CSV path')
    parser.add_argument('--no-standardize', action='store_true',
                        help='Skip compound standardization')

    args = parser.parse_args()

    if 'all' in args.models:
        args.models = MODEL_NAMES
    else:
        unknown = set(args.models) - set(MODEL_NAMES)
        if unknown:
            print(f"[ERROR] Unknown models: {unknown}")
            print(f"  Available: {MODEL_NAMES}")
            sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    if not check_deepmol():
        sys.exit(1)
    if not check_deepmol_models():
        sys.exit(1)

    from deepmol.loaders.loaders import CSVLoader
    from deepmol.standardizer import BasicStandardizer, CustomStandardizer

    # --- Load ---
    print(f"\n[1/3] Loading {args.input}...")
    loader = CSVLoader(
        dataset_path=args.input,
        smiles_field=args.smiles_col,
        id_field=args.id_col or args.smiles_col,
        labels_fields=[],
        mode='auto'
    )
    dataset = loader.create_dataset()
    n, _ = dataset.get_shape()
    print(f"  Loaded {n} molecules")

    # --- Standardize ---
    if not args.no_standardize:
        print("[2/3] Standardizing compounds...")
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
        print(f"  {n - n_after[0]} molecules removed, {n_after[0]} remaining")
    else:
        print("[2/3] Skipping standardization")

    # --- Predict ---
    print(f"[3/3] Running ADMET predictions ({len(args.models)} models)...")
    ids = dataset.ids
    results = {'id': ids}

    for model_name in args.models:
        print(f"  Running {model_name} model...")
        try:
            # Dynamic import of model class
            import importlib
            module = importlib.import_module('deepmol_models')
            ModelClass = getattr(module, f'{model_name}Model')
            model = ModelClass()

            predictions = model.predict(dataset)
            # Flatten if needed
            if hasattr(predictions, 'flatten'):
                predictions = predictions.flatten()
            results[model_name] = predictions

            # Stats
            vals = np.array(predictions)
            if len(vals.shape) <= 1:
                print(f"    Predicted {len(vals)} values, range: [{vals.min():.4f}, {vals.max():.4f}]")
        except Exception as e:
            print(f"    [ERROR] {model_name} failed: {e}")
            results[model_name] = ['ERROR'] * len(ids)

    # --- Save ---
    print(f"\nSaving results to {args.output}...")
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"  Saved {len(df)} rows x {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  File: {os.path.abspath(args.output)}")
    print("Done.")


if __name__ == '__main__':
    main()
