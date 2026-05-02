#!/usr/bin/env python3
"""
Batch Molecular Featurization Script

Featurizes a set of molecules (CSV, SDF, or plain SMILES) and saves
the feature matrix as a CSV or NumPy file.

Usage:
    python batch_featurization.py molecules.csv --smiles-col SMILES --featurizer morgan --output features.csv
    python batch_featurization.py molecules.smi --smiles-col 0 --featurizer maccs --output features.npy
    python batch_featurization.py molecules.sdf --featurizer morgan --standardize chembl

Output formats: .csv (with headers), .npy (NumPy array)
"""

import argparse
import os
import sys

import numpy as np


def check_deepmol():
    try:
        import deepmol
        print(f"[OK] DeepMol {deepmol.__version__}")
        return True
    except ImportError:
        print("[ERROR] DeepMol not installed. Run: pip install deepmol[all]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch featurize molecules with DeepMol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_featurization.py library.csv --smiles-col SMILES --featurizer morgan
  python batch_featurization.py molecules.sdf --featurizer maccs --standardize chembl
  python batch_featurization.py compounds.smi --smiles-col 0 --featurizer morgan --fp-size 1024 --radius 3
        """
    )
    parser.add_argument('input', help='Input file (CSV, SDF, or .smi)')
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES column name')
    parser.add_argument('--id-col', default=None, help='ID column name')
    parser.add_argument('--featurizer', default='morgan',
                        choices=['morgan', 'maccs', 'rdk', 'atompair', 'layered', 'mol2vec'])
    parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint bit size')
    parser.add_argument('--radius', type=int, default=2, help='Morgan radius (2=ECFP4, 3=ECFP6)')
    parser.add_argument('--standardize', default='heavy',
                        choices=['none', 'basic', 'heavy', 'chembl'])
    parser.add_argument('--output', default='features.csv', help='Output file path')
    parser.add_argument('--include-ids', action='store_true',
                        help='Include molecule IDs in CSV output')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    if not check_deepmol():
        sys.exit(1)

    from deepmol.loaders.loaders import CSVLoader
    from deepmol.loaders import SDFLoader
    from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer
    from deepmol.compound_featurization import (
        MorganFingerprint, MACCSkeysFingerprint, RDKFingerprint,
        AtomPairFingerprint, LayeredFingerprint
    )

    # --- 1. Load ---
    ext = os.path.splitext(args.input)[1].lower()
    print(f"[1/4] Loading {args.input}...")

    if ext in ('.sdf', '.mol'):
        loader = SDFLoader(
            dataset_path=args.input,
            id_field=args.id_col or 'id',
            labels_fields=[],
            mode='auto'
        )
    else:
        loader = CSVLoader(
            dataset_path=args.input,
            smiles_field=args.smiles_col,
            id_field=args.id_col or args.smiles_col,
            labels_fields=[],
            mode='auto'
        )

    dataset = loader.create_dataset()
    n_samples, _ = dataset.get_shape()
    print(f"  Loaded {n_samples} molecules")

    # --- 2. Standardize ---
    if args.standardize == 'none':
        print("[2/4] Skipping standardization")
    else:
        print(f"[2/4] Standardizing (method={args.standardize})...")
        if args.standardize == 'basic':
            BasicStandardizer().standardize(dataset, inplace=True)
        elif args.standardize == 'chembl':
            ChEMBLStandardizer().standardize(dataset, inplace=True)
        else:  # heavy
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
        print(f"  {n_samples - n_after[0]} molecules removed")

    # --- 3. Featurize ---
    print(f"[3/4] Generating {args.featurizer} fingerprints...")
    if args.featurizer == 'morgan':
        featurizer = MorganFingerprint(radius=args.radius, size=args.fp_size)
    elif args.featurizer == 'maccs':
        featurizer = MACCSkeysFingerprint()
    elif args.featurizer == 'rdk':
        featurizer = RDKFingerprint(fpSize=args.fp_size)
    elif args.featurizer == 'atompair':
        featurizer = AtomPairFingerprint(nBits=args.fp_size)
    elif args.featurizer == 'layered':
        featurizer = LayeredFingerprint(fpSize=args.fp_size)
    elif args.featurizer == 'mol2vec':
        from deepmol.compound_featurization import Mol2Vec
        featurizer = Mol2Vec()

    featurizer.featurize(dataset, inplace=True)
    X = dataset.X
    ids = dataset.ids
    print(f"  Feature matrix shape: {X.shape}")

    # --- 4. Save ---
    print(f"[4/4] Saving to {args.output}...")
    out_ext = os.path.splitext(args.output)[1].lower()

    if out_ext == '.npy':
        np.save(args.output, X)
    else:
        # CSV with/without IDs
        if args.include_ids and ids is not None:
            header = ['id'] + [f'fp_{i}' for i in range(X.shape[1])]
            data = np.column_stack([np.array(ids).reshape(-1, 1), X])
        else:
            header = [f'fp_{i}' for i in range(X.shape[1])]
            data = X
        np.savetxt(args.output, data, delimiter=',', header=','.join(header),
                   fmt='%s', comments='')

    print(f"  Saved {X.shape[0]} x {X.shape[1]} matrix to {os.path.abspath(args.output)}")
    print("Done.")


if __name__ == '__main__':
    main()
