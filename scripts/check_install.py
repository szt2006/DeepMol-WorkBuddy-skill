#!/usr/bin/env python3
"""
DeepMol Installation Checker

Verifies that DeepMol and all its dependencies are correctly installed
and importable. Run after installation to confirm everything works.

Usage:
    python check_install.py
"""

import sys
import os


def green(text): return f"\033[92m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def bold(text): return f"\033[1m{text}\033[0m"


def check_python():
    """Check Python version compatibility."""
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 13):
        return True, f"Python {major}.{minor} (may need step-by-step install, not 'pip install deepmol[all]')"
    elif (major, minor) >= (3, 9):
        return True, f"Python {major}.{minor}"
    else:
        return False, f"Python {major}.{minor} (too old, need >= 3.9)"


def check_env_vars():
    """Check required env vars for Windows sandbox."""
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    if user:
        return True, f"USER={user}"
    return False, "USER/USERNAME not set (may cause torch import failure)"


def try_import(name, package=None):
    """Try to import a module. Returns (ok, message)."""
    try:
        mod = __import__(name)
        version = getattr(mod, '__version__', 'installed')
        return True, f"{name} {version}"
    except ImportError as e:
        msg = str(e).split('\n')[0]
        hint = f" (pip install {package or name})" if package or not package else ""
        return False, f"{name} NOT FOUND: {msg}{hint}"


def main():
    print("=" * 60)
    print("  DeepMol Installation Checker")
    print("=" * 60)

    results = []

    # Python
    print("\n--- System ---")
    checks = [("Python", check_python), ("Env vars", check_env_vars)]
    for label, check in checks:
        ok, msg = check()
        status = "[OK]" if ok else "[FAIL]"
        col = green if ok else red
        print(f"  {col(status)} {msg}")
        results.append(("System", label, ok))

    # Core dependencies
    print("\n--- Core Dependencies ---")
    core_pkgs = [
        ("numpy", None), ("pandas", None), ("scipy", None),
        ("sklearn", "scikit-learn"), ("matplotlib", None),
        ("torch", None),
    ]
    for name, pkg in core_pkgs:
        ok, msg = try_import(name, pkg)
        status = "[OK]" if ok else "[FAIL]"
        color = green if ok else red
        print(f"  {color(status)} {msg}")
        results.append(("Core dep", name, ok))

    # DeepMol-specific
    print("\n--- DeepMol Core ---")
    deepmol_modules = [
        ("deepmol", "deepmol"),
        ("deepmol.loaders.loaders", None),
        ("deepmol.standardizer", None),
        ("deepmol.compound_featurization", None),
        ("deepmol.feature_selection", None),
        ("deepmol.splitters.splitters", None),
        ("deepmol.models.sklearn_models", None),
        ("deepmol.metrics.metrics", None),
        ("deepmol.unsupervised", None),
        ("deepmol.imbalanced_learn.imbalanced_learn", None),
        ("deepmol.feature_importance", None),
        ("deepmol.pipeline", None),
        ("deepmol.pipeline_optimization", None),
        ("deepmol.parameter_optimization.hyperparameter_optimization", None),
    ]
    for modname, pkg in deepmol_modules:
        ok, msg = try_import(modname, pkg)
        status = "[OK]" if ok else "[FAIL]"
        color = green if ok else red
        print(f"  {color(status)} {msg}")
        results.append(("DeepMol", modname.split('.')[-1], ok))

    # Optional
    print("\n--- Optional ---")
    optional = [
        ("optuna", None, "Pipeline optimization"),
        ("rdkit", None, "Molecular processing"),
        ("shap", None, "Model explainability"),
        ("h5py", None, "HDF5 model saving"),
        ("plotly", None, "Interactive plots"),
        ("transformers", None, "LLM-based featurization"),
    ]
    for name, pkg, purpose in optional:
        ok, msg = try_import(name, pkg)
        status = "[OK]" if ok else "[SKIP]"
        color = green if ok else yellow
        print(f"  {color(status)} {msg} ({purpose})")
        results.append(("Optional", name, ok))

    # JAX check (must NOT be installed)
    print("\n--- Conflict Check ---")
    jax_ok, jax_msg = try_import("jax")
    if not jax_ok:
        print(f"  {green('[OK]')} JAX not installed (good)")
    else:
        print(f"  {red('[WARN]')} JAX is installed -- may cause conflicts! pip uninstall jax jaxlib")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, _, ok in results if ok)
    total = len(results)
    all_ok = passed == total

    if all_ok:
        print(f"  {green(bold('ALL CHECKS PASSED'))} ({passed}/{total})")
        print("  DeepMol is ready to use!")
    else:
        failed = total - passed
        print(f"  {passed}/{total} passed, {red(str(failed) + ' failed')}")
        print()
        print("  Failed components:")
        for cat, name, ok in results:
            if not ok:
                print(f"    [{cat}] {name}")
        print()
        print("  Common fixes:")
        print("    pip install scikit-learn")
        print("    pip install rdkit pandas seaborn ...  (see SKILL.md Installation section)")
        if not os.environ.get('USER') and not os.environ.get('USERNAME'):
            print("    Set USER environment variable for torch: set USER=Administrator")

    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
