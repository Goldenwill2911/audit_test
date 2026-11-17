"""Generate a synthetic readmission dataset with repeated encounters per patient.

Each patient has a stable latent risk that drives both their features and their
readmission label. A model that sees the same patient in training and test can
memorize that latent signal, which is exactly what patient-level leakage exploits.
No real patient data is used.
"""
import numpy as np
import pandas as pd

RNG_SEED = 7
N_PATIENTS = 2500


def main(out_path="data/encounters.csv"):
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for pid in range(N_PATIENTS):
        latent = rng.normal()                       # patient-specific risk
        fingerprint = rng.normal(size=4)            # stable per-patient quirks
        idio = rng.normal(0, 1.8)                   # per-patient risk not derivable from features
        for enc in range(rng.integers(1, 7)):       # 1-6 encounters
            age = 40 + 10 * fingerprint[0] + rng.normal(0, 1)
            los = np.clip(4 + 2.0 * latent + rng.normal(0, 1), 0, None)
            prior_adm = max(0, int(round(1 + latent + rng.normal(0, 1))))
            lab_a = 1.0 * fingerprint[1] + 0.3 * latent + rng.normal(0, 0.1)
            lab_b = 1.0 * fingerprint[2] + rng.normal(0, 0.1)
            lab_c = 1.0 * fingerprint[3] + rng.normal(0, 0.1)
            logit = -0.8 + 1.6 * latent + 0.15 * rng.normal()
            # Part of the label is a per-patient offset unrelated to the features.
            # Only a model that has seen this patient (via the fingerprint) can recover it.
            logit += idio
            label = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(dict(patient_id=f"P{pid:05d}", encounter_id=f"E{pid:05d}_{enc}",
                             age=age, length_of_stay=los, prior_admissions=prior_adm,
                             lab_a=lab_a, lab_b=lab_b, lab_c=lab_c, readmit_30d=label))
    df = pd.DataFrame(rows)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} encounters for {df.patient_id.nunique()} patients -> {out_path}")


if __name__ == "__main__":
    main()
