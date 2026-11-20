# Leakage Audit Report (Demonstration on Synthetic Data)

All data is synthetic. The audited pipeline is a deliberately flawed stand-in.

## Integrity

- Audited tree unchanged after audit: **True**
- Data SHA-256: `c37b88a78222f06b4a0001f3a0d46eba942f2116a320c7aaecf59b998cf6adae`
- Environment: Python 3.13.18, scikit-learn 1.9.1

## Headline

- Reported AUROC (reproduced, deterministic=True): 0.736
- Patient-disjoint AUROC: 0.696 +/- 0.016
- Test patients also in training: 92.8%

## Controlled ablation (5-fold CV in every arm)

| Split | Preprocessing | AUROC |
|---|---|---|
| encounter | prefit | 0.736 |
| encounter | infold | 0.736 |
| patient | prefit | 0.696 |
| patient | infold | 0.696 |

## Findings

### LEAK-001 [Critical] Train/test split is by encounter, not patient

- Status: Verified
- Evidence: 92.8% of test patients (1164/1254) also appear in training.
- Reproduce: `python audit/harness.py  (see overlap metric in findings.json)`

### LEAK-002 [Low] Imputer and scaler are fit before the split

- Status: Verified
- Evidence: preprocess() output has zero mean over all rows, so test rows informed the fitted statistics. Max mean shift vs train-only fit: 0.0305. Measured AUROC impact under controlled ablation: 0.0000. Random forests are scale-invariant and the data has no missing values, so impact is negligible here; it would rise for scale-sensitive models or uneven missingness.
- Reproduce: `python audit/harness.py`

### EVAL-001 [Critical] Reported AUROC is inflated relative to patient-disjoint evaluation

- Status: Verified
- Evidence: Reported 0.736 vs patient-grouped 5-fold 0.696 (SD 0.016). Gap exceeds 2 SD of fold variance.
- Reproduce: `python audit/harness.py`
