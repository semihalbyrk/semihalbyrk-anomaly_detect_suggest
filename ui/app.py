# ----------------------------- app.py (v4.1) -----------------------------
"""
TexNL Service-Point Dashboard
– Yan-yana KPI (Total / Anomaly)  ⚬  no delta
– CAIv histogramı
– CSV download
– Kırmızı satır: Anomaly = Yes
-------------------------------------------------
Run:
    streamlit run ui/app.py
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

SP_CSV = Path("output/sp_metrics.csv")

# ---------- page ----------
st.set_page_config("TexNL Dashboard", layout="wide")
st.title("Service Point Metrics")

# ---------- load ----------
if not SP_CSV.exists():
    st.error("sp_metrics.csv bulunamadı – infer.py çalıştırın.")
    st.stop()
df = pd.read_csv(SP_CSV)

# ---------- KPI ----------
tot  = len(df)
anom = (df["Anomaly State"] == "Yes").sum()

k1, k2 = st.columns(2)
k1.metric("Total SP", f"{tot:,}")
k2.metric("Anomaly SP", f"{anom:,}")

# ---------- histogram ----------
with st.expander("CAIv Distribution", expanded=False):
    hist = (
        alt.Chart(df, height=120)
        .mark_bar(opacity=0.7)
        .encode(
            alt.X("CAIv Ratio", bin=alt.Bin(maxbins=30)),
            y="count()"
        )
    )
    st.altair_chart(hist, use_container_width=True)

st.divider()

# ---------- filter ----------
show_only = st.checkbox("Show only anomalies", False)
view = df[df["Anomaly State"] == "Yes"] if show_only else df

# ---------- table style ----------
def row_color(r):
    color = "#ffe6e6" if r["Anomaly State"] == "Yes" else "white"
    return [f"background-color:{color}" for _ in r]

st.dataframe(
    view.style.apply(row_color, axis=1),
    use_container_width=True,
    height=550,
)

# ---------- download ----------
st.download_button(
    "Download current view as CSV",
    data=view.to_csv(index=False).encode(),
    file_name="sp_metrics_view.csv",
    mime="text/csv",
)


# ---------- PIPELINE EXPLANATION ------------------------------------------------
st.markdown(
    r"""
<details>
<summary style="font-size:18px;font-weight:700;">ℹ️ About the pipeline – 5 stages & Isolation Forest rationale</summary>

### Sequential stages

| # | Stage | What happens? | Manual choice? |
|---|-------|---------------|----------------|
| **1** | **ETL → visit table** | Raw Excel → **one row per collection visit**.<br>Derived columns: `V_kg` · `capacity_kg` · `V_fill = V_kg/cap` · `VI` (days since prev. visit) · `GR = V_kg/VI` · 6-day rolling mean / std. | — |
| **2** | **Unsupervised ML score** | **Isolation Forest** fits those visit rows *without labels* and assigns each row a continuous **anomaly score (0 → ∞)**.<br>Lower *isolation depth* ⇒ higher score. | Model hyper-parameters:<br>`n_estimators = 400`<br>`max_samples = "auto"` |
| **3** | **Threshold → row is_anomaly** | Choose where to cut the score distribution: `contamination = 0.05` ⇒ “top 5 % of scores are anomalies”. | Yes – change 0.05 to 0.03, 0.10… |
| **4** | **Aggregate to service-point** | • Take **max anomaly_score** of its visits (worst day).<br>• Compute 10 business KPIs (CAIv, VOF %, …).<br>• Mark **“Anomaly State = Yes”** if that max score is within the top 5 % of all SPs. | Same contamination sets the 5 % cut-off. |
| **5** | **Dashboard** | This Streamlit page: KPI cards · CAIv histogram · filter & red-row table · CSV download · this explanation panel. | — |

---

### Why Isolation Forest?

* **Unsupervised** – no labelled “bad” examples needed.  
* **Density-aware** – captures both global outliers and local micro-clusters.  
* **Linear scalability** – thousands of visits processed in milliseconds.

**Intuition**  
Randomly partition the feature space with many binary trees.  
Points that become isolated in **fewer splits** live in sparse regions ⇒ **higher anomaly score**.

</details>
""",
    unsafe_allow_html=True
)
# ---------------------------------------------------------------------------
