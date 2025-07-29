# ----------------------------- utils.py -----------------------------
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_features(pq_path: str, fit_scaler: bool = False):
    """
    Parquet dosyasını okur, 'service_point' & 'visit_date' dışındaki
    sütunları özellik matrisi olarak döndürür; aynı anda NaN'leri
    sütun medyanıyla doldurur ve StandardScaler uygular.

    Parameters
    ----------
    pq_path : str
        Parquet dosya yolu (visit-level özellik matrisi).
    fit_scaler : bool, default False
        True ise scaler'ı bu X üzerinde fit eder ve .scaler.pkl dosyası oluşturur.

    Returns
    -------
    df : pandas.DataFrame
        Orijinal DataFrame (NaN'ler doldurulmuş).
    X_scaled : numpy.ndarray
        Ölçeklenmiş özellik matrisi.
    """
    pq_path = Path(pq_path)
    df = pd.read_parquet(pq_path)

    feature_cols = [c for c in df.columns if c not in ("service_point", "visit_date")]

    # ---- NaN → sütun medyanı
    med = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(med)

    X = df[feature_cols].values

    scaler_path = pq_path.with_suffix(".scaler.pkl")
    if fit_scaler or not scaler_path.exists():
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X)
    return df, X_scaled
