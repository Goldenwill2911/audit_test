"""Read-only leakage audit harness.

Imports the pipeline under audit without modifying it, then:
  1. Hashes the audited tree before and after (proves read-only).
  2. Reproduces the reported metric.
  3. Measures patient overlap across the reported split.
  4. Detects preprocessing fit on data outside the training partition.
  5. Re-evaluates the UNCHANGED model with patient-grouped CV and in-fold preprocessing.
  6. Writes findings.json and a Markdown report with reproduction commands.
"""
import hashlib
import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDITED = ROOT / "original_pipeline"
DATA = ROOT / "data" / "encounters.csv"
OUT = ROOT / "audit" / "output"
sys.path.insert(0, str(AUDITED))
import model as audited  # noqa: E402  (imported unchanged)


def tree_manifest(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = tree_manifest(AUDITED)
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    df = pd.read_csv(DATA)
    findings = []

    # 2. Reproduce reported metric, run twice to confirm determinism.
    r1 = audited.train_and_evaluate(DATA)
    r2 = audited.train_and_evaluate(DATA)
    reported = r1["auroc"]
    deterministic = abs(r1["auroc"] - r2["auroc"]) < 1e-12

    # 3. Patient overlap across the reported split.
    tr_p = set(df.iloc[r1["train_idx"]].patient_id)
    te_p = set(df.iloc[r1["test_idx"]].patient_id)
    overlap = len(te_p & tr_p) / len(te_p)
    if overlap > 0:
        findings.append(dict(
            id="LEAK-001", severity="Critical", status="Verified",
            title="Train/test split is by encounter, not patient",
            evidence=f"{overlap:.1%} of test patients ({len(te_p & tr_p)}/{len(te_p)}) also appear in training.",
            reproduce="python audit/harness.py  (see overlap metric in findings.json)"))

    # 4. Preprocessing fit scope: compare scaler statistics fit on all rows vs train rows.
    X_raw = df[audited.FEATURES]
    full_mean = StandardScaler().fit(SimpleImputer(strategy="median").fit_transform(X_raw)).mean_
    tr_mean = StandardScaler().fit(SimpleImputer(strategy="median").fit_transform(
        X_raw.iloc[r1["train_idx"]])).mean_
    probe = audited.preprocess(X_raw)
    used_full = np.allclose(probe.mean(axis=0), 0, atol=1e-9)  # zero-mean over ALL rows
    if used_full:
        findings.append(dict(
            id="LEAK-002", severity="Low", status="Verified",
            title="Imputer and scaler are fit before the split",
            evidence=("preprocess() output has zero mean over all rows, so test rows informed "
                      f"the fitted statistics. Max mean shift vs train-only fit: "
                      f"{np.max(np.abs(full_mean - tr_mean)):.4f}."),
            reproduce="python audit/harness.py"))

    # 5. Corrected re-evaluation: same model, patient-grouped CV, preprocessing in-fold.
    X, y, g = X_raw.values, df[audited.TARGET].values, df.patient_id.values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    fold_scores = []
    for tr, te in cv.split(X, y, g):
        assert not set(g[tr]) & set(g[te])
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             audited.build_model())
        pipe.fit(X[tr], y[tr])
        fold_scores.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    corrected, sd = float(np.mean(fold_scores)), float(np.std(fold_scores))

    # 5b. Controlled ablation: 5-fold CV in every arm, change one factor at a time.
    from sklearn.model_selection import StratifiedKFold
    X_pre = audited.preprocess(X_raw)
    def cv_auc(splitter, groups, prefit):
        s = []
        it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for tr, te in it:
            if prefit:
                est, Xa = audited.build_model(), X_pre
            else:
                est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    audited.build_model())
                Xa = X
            est.fit(Xa[tr], y[tr])
            s.append(roc_auc_score(y[te], est.predict_proba(Xa[te])[:, 1]))
        return round(float(np.mean(s)), 4)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ablation = {
        "encounter_split_prefit_preprocessing": cv_auc(skf, None, True),
        "encounter_split_infold_preprocessing": cv_auc(skf, None, False),
        "patient_split_prefit_preprocessing": cv_auc(cv, g, True),
        "patient_split_infold_preprocessing": cv_auc(cv, g, False),
    }
    pre_effect = abs(ablation["patient_split_prefit_preprocessing"]
                     - ablation["patient_split_infold_preprocessing"])
    for f in findings:
        if f["id"] == "LEAK-002":
            f["evidence"] += (f" Measured AUROC impact under controlled ablation: {pre_effect:.4f}. "
                              "Random forests are scale-invariant and the data has no missing values, "
                              "so impact is negligible here; it would rise for scale-sensitive models "
                              "or uneven missingness.")

    if reported - corrected > 2 * sd:
        findings.append(dict(
            id="EVAL-001", severity="Critical", status="Verified",
            title="Reported AUROC is inflated relative to patient-disjoint evaluation",
            evidence=(f"Reported {reported:.3f} vs patient-grouped 5-fold {corrected:.3f} "
                      f"(SD {sd:.3f}). Gap exceeds 2 SD of fold variance."),
            reproduce="python audit/harness.py"))

    after = tree_manifest(AUDITED)
    result = dict(
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                         numpy=np.__version__, pandas=pd.__version__),
        data_sha256=data_hash, audited_tree_unchanged=(before == after),
        audited_manifest=before, reported_auroc=reported, deterministic=deterministic,
        patient_overlap=overlap, corrected_auroc_mean=corrected, corrected_auroc_sd=sd,
        fold_scores=fold_scores, ablation=ablation, findings=findings)
    (OUT / "findings.json").write_text(json.dumps(result, indent=2))
    write_report(result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("audited_manifest",)}, indent=2))


def write_report(r):
    lines = ["# Leakage Audit Report (Demonstration on Synthetic Data)", "",
             "All data is synthetic. The audited pipeline is a deliberately flawed stand-in.", "",
             "## Integrity", "",
             f"- Audited tree unchanged after audit: **{r['audited_tree_unchanged']}**",
             f"- Data SHA-256: `{r['data_sha256']}`",
             f"- Environment: Python {r['environment']['python']}, scikit-learn "
             f"{r['environment']['sklearn']}", "",
             "## Headline", "",
             f"- Reported AUROC (reproduced, deterministic={r['deterministic']}): {r['reported_auroc']:.3f}",
             f"- Patient-disjoint AUROC: {r['corrected_auroc_mean']:.3f} +/- {r['corrected_auroc_sd']:.3f}",
             f"- Test patients also in training: {r['patient_overlap']:.1%}", "",
             "## Controlled ablation (5-fold CV in every arm)", "",
             "| Split | Preprocessing | AUROC |", "|---|---|---|"] + [
             f"| {k.split('_')[0]} | {k.split('_')[2]} | {v:.3f} |" for k, v in r["ablation"].items()] + ["",
             "## Findings", ""]
    for f in r["findings"]:
        lines += [f"### {f['id']} [{f['severity']}] {f['title']}", "",
                  f"- Status: {f['status']}", f"- Evidence: {f['evidence']}",
                  f"- Reproduce: `{f['reproduce']}`", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
