"""Stand-in for the 'frozen' code under audit. Contains deliberate, realistic flaws.

The audit harness must NOT modify this file. It imports it unchanged.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "length_of_stay", "prior_admissions", "lab_a", "lab_b", "lab_c"]
TARGET = "readmit_30d"


def build_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                  random_state=0, n_jobs=-1)


def preprocess(X):
    # FLAW 2: imputer and scaler are fit on the full dataset before splitting.
    X = SimpleImputer(strategy="median").fit_transform(X)
    return StandardScaler().fit_transform(X)


def train_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    X = preprocess(df[FEATURES])
    y = df[TARGET].values
    # FLAW 1: split by shuffled encounter rows, not by patient.
    idx = range(len(df))
    tr, te = train_test_split(list(idx), test_size=0.2, random_state=42, stratify=y)
    model = build_model().fit(X[tr], y[tr])
    auroc = roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])
    return {"auroc": auroc, "train_idx": tr, "test_idx": te}


if __name__ == "__main__":
    print(train_and_evaluate("data/encounters.csv")["auroc"])
