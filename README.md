# Read-Only Leakage Audit Harness (Demonstration)

A small, reproducible demonstration of how I audit a Python clinical ML pipeline
without modifying it. Everything here is synthetic. No real patient data, and no
client code, is included.

## What it shows

`original_pipeline/model.py` is a deliberately flawed stand-in for frozen code under
audit. It has two common defects:

1. The train/test split shuffles encounters, so the same patient lands in both sets.
2. The imputer and scaler are fit on the full dataset before the split.

`audit/harness.py` imports that pipeline **unchanged** and:

- hashes the audited tree before and after the run to prove it stays read-only
- reproduces the reported AUROC and checks that it's deterministic
- measures patient overlap across the reported split
- detects preprocessing that was fit outside the training partition
- re-evaluates the same model with patient-grouped 5-fold CV and preprocessing fit inside each fold
- writes `findings.json` and `REPORT.md`, with a severity, status, evidence and reproduction command for each finding

## Run

```bash
python make_synthetic_data.py
python audit/harness.py
```

Requires Python 3.10+, numpy, pandas, scikit-learn.

## Result on the included synthetic data

| Measure | Value |
|---|---|
| Reported AUROC (encounter-level split) | 0.736 |
| Patient-disjoint AUROC (5-fold, SD) | 0.696 (0.016) |
| Test patients also in training | 92.8% |
| Audited tree unchanged | True |

The synthetic generator gives each patient a risk offset that has nothing to do with
their features. A model can only recover it by recognizing a patient it has already
seen, which is why the inflation appears only when patients leak across the split.
# audit_test
