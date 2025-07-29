# ----------------------------- infer.py (v4) -----------------------------
"""
Üretir: output/sp_metrics.csv   (tek satır / Service Point)

Sütun adları (gösterim-dostu)
-----------------------------
Service Point
Visit Count
Max Anomaly Score
CAIv Ratio
VOF %
VUR %
CVv Ratio
PMRv Ratio
GR p90 (kg/day)
DtO (days)
IG (days)
CVgr Ratio
Anomaly State      (Yes / No)
"""
import argparse, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from utils import load_features
from scipy.stats import zscore   # sadece NaN kontrolünde kullanılıyor

def score_visits(cfg, in_pq):
    mdl = joblib.load(cfg["paths"]["model_out"])
    df_vis, X = load_features(in_pq, fit_scaler=False)

    df_vis["anomaly_score"] = -mdl.score_samples(X)
    q = 1 - cfg["iforest"]["contamination"]
    thr = np.quantile(df_vis["anomaly_score"], q)
    df_vis["is_anomaly"] = (df_vis["anomaly_score"] >= thr).astype(int)
    return df_vis

def build_sp(df_vis: pd.DataFrame, contamination: float) -> pd.DataFrame:
    g = df_vis.groupby("service_point")

    df = pd.DataFrame({
        "Service Point": g.size().index,
        "Visit Count":   g.size().values,
        "Max Anomaly Score": g["anomaly_score"].max().values,
        "CAIv Ratio":    g["V_kg"].quantile(0.90) / g["capacity_kg"].first(),
        "VOF %":         g["V_fill"].apply(lambda s: (s > 1).mean()*100),
        "VUR %":         g["V_fill"].mean()*100,
        "CVv Ratio":     g["V_kg"].std() / g["V_kg"].mean(),
        "PMRv Ratio":    g["V_kg"].max() / g["V_kg"].mean(),
        "GR p90 (kg/day)": g["GR"].quantile(0.90),
        "DtO (days)":      g["capacity_kg"].first() / g["GR"].median(),
        "IG (days)":       g["VI"].max(),
        "CVgr Ratio":      g["GR"].std() / g["GR"].mean(),
    })

    # ML-tabanlı anomaly etiketi
    q = np.quantile(df["Max Anomaly Score"], 1 - contamination)
    df["Anomaly State"] = np.where(df["Max Anomaly Score"] >= q, "Yes", "No")

    return df

def main(cfg, in_pq):
    cfg = yaml.safe_load(open(cfg))
    df_vis = score_visits(cfg, in_pq)

    sp_df = build_sp(df_vis, cfg["iforest"]["contamination"])
    Path("output").mkdir(exist_ok=True)
    sp_df.to_csv("output/sp_metrics.csv", index=False)
    print("✅  output/sp_metrics.csv yazıldı")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",   default="src/config.yml")
    p.add_argument("--in_pq", default="data/processed/visits.parquet")
    a = p.parse_args()
    main(a.cfg, a.in_pq)
