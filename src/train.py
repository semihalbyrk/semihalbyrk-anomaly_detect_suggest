import argparse, yaml, joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from utils import load_features

def train(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    df, X = load_features(
        cfg["paths"]["train_matrix"],  # fit_scaler=True ⟹ scalerı da kaydeder
        fit_scaler=True
    )

    mdl = IsolationForest(
        n_estimators = cfg["iforest"]["n_estimators"],
        contamination= cfg["iforest"]["contamination"],
        bootstrap    = True,
        random_state = 42,
        n_jobs       = -1
    )
    mdl.fit(X)
    Path(cfg["paths"]["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, cfg["paths"]["model_out"])
    print("✅ Model saved →", cfg["paths"]["model_out"])

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--cfg", default="src/config.yml")
    args = a.parse_args()
    train(args.cfg)
