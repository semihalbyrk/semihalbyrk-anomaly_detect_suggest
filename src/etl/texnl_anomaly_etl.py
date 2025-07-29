"""
TexNL ETL + Feature Engineering  (v1.1)
---------------------------------------
⟶Robust kolon adi eşleştirmeleri eklenmiştir.

Input  : Excel dosyasi (Task Record, Service Points, Assets)
Output : data/processed/visits.parquet (ziyaret + özellikler)

Usage:
python src/etl/texnl_anomaly_etl.py \
       --input data/raw/TexNL_Data.xlsx \
       --out   data/processed/visits.parquet
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------
COL_SP_CANDIDATES_TASK   = {"Service Point", "Service Point Name", "service_point"}
COL_DATE_CANDIDATES_TASK = {"Date", "Task Date", "visit_date"}

COL_SP_CANDIDATES_SP     = {"Name", "Service Point", "service_point"}
COL_SP_CANDIDATES_ASSET  = {"Location Details", "Service Point", "service_point"}


def find_column(df: pd.DataFrame, candidates: set[str]) -> str:
    """Return first matching column name (case‑insensitive)."""
    for c in df.columns:
        if c.strip() in candidates:
            return c
    raise KeyError(f"Hiçbir kolon bulunamadı; aranan seçenekler: {candidates}")


def compute_interval_kpi(df_visits: pd.DataFrame) -> pd.DataFrame:
    df_visits = df_visits.sort_values(["service_point", "visit_date"])
    df_visits["VI"] = (
        df_visits.groupby("service_point")["visit_date"]
        .diff().dt.days.fillna(np.nan)
    )
    df_visits["GR"] = df_visits["V_kg"] / df_visits["VI"]
    return df_visits

# --------------------------------------------------
# Ana ETL fonksiyonu
# --------------------------------------------------

def run_etl(input_xlsx: str, out_pq: str):
    xlsx = pd.ExcelFile(input_xlsx, engine="openpyxl")

    # Task sheet ------------------------------------------------------------
    tasks = pd.read_excel(xlsx, sheet_name="Task Record")

    col_sp_task   = find_column(tasks, COL_SP_CANDIDATES_TASK)
    col_date_task = find_column(tasks, COL_DATE_CANDIDATES_TASK)

    tasks = tasks.rename(columns={col_sp_task: "service_point", col_date_task: "visit_date"})

    # Sadece Bag Weight satırları
    tasks = tasks[tasks["Material"].str.contains("Bag Weight", na=False)].copy()
    tasks["visit_date"] = pd.to_datetime(tasks["visit_date"]).dt.date
    tasks["V_kg"] = tasks["Actual Amount (Item)"].astype(float)

    # Assets sheet ----------------------------------------------------------
    assets = pd.read_excel(xlsx, sheet_name="Assets")
    col_sp_asset = find_column(assets, COL_SP_CANDIDATES_ASSET)
    assets = assets.rename(columns={col_sp_asset: "service_point"})
    assets["capacity_kg"] = assets["Weight Capacity"].astype(float)
    cap = (
        assets.groupby("service_point", as_index=False)["capacity_kg"].sum()
    )

    # Merge & aggregate ------------------------------------------------------
    df = (
        tasks.groupby(["service_point", "visit_date"], as_index=False)["V_kg"].sum()
        .merge(cap, how="left", on="service_point")
        .dropna(subset=["capacity_kg"])
    )

    df["V_fill"] = df["V_kg"] / df["capacity_kg"]
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df = compute_interval_kpi(df)

    # Basit rolling özellikler ------------------------------------------------
    df = df.sort_values(["service_point", "visit_date"])
    df["V_kg_mean"] = (
        df.groupby("service_point")["V_kg"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    )
    df["V_kg_std"] = (
        df.groupby("service_point")["V_kg"].transform(lambda s: s.rolling(6, min_periods=1).std().fillna(0))
    )

    # Kaydet -----------------------------------------------------------------
    Path(out_pq).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_pq, index=False)
    print(f"✅ visits parquet written → {out_pq}  |  {len(df):,} satır")


# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_etl(args.input, args.out)
