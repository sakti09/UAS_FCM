import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Clustering", layout="wide")

# keterangan: load CSS global theme (insight.css)
def load_css(path: str = "assets/insight.css"):
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

load_css()

st.title("Page 2 — Clustering Dashboard")
st.caption("Halaman segmentasi pelanggan berdasarkan hasil clustering. Input wajib berupa CSV hasil clustering.")

# keterangan: upload CSV (wajib)
uploaded = st.file_uploader("Upload CSV (hasil clustering)", type=["csv"])
if not uploaded:
    st.info("Upload file CSV hasil clustering untuk menampilkan dashboard.")
    st.stop()

df = pd.read_csv(uploaded)

# -------------------------
# Helpers
# -------------------------
def pick_col(df_in, candidates):
    lower_map = {c.lower(): c for c in df_in.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"

def fmt_money(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"

def render_kpi_cluster(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="kpi-cluster">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def smart_xtick_rotation(values) -> int:
    vals = [str(v) for v in values]
    if not vals:
        return 0
    maxlen = max(len(v) for v in vals)
    n = len(vals)
    return 45 if (maxlen >= 12 or n >= 8) else 0

def apply_filters(df_in: pd.DataFrame, filter_state: dict) -> pd.DataFrame:
    out = df_in.copy()
    for col, sel in filter_state.items():
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            lo, hi = sel
            out = out[pd.to_numeric(out[col], errors="coerce").between(lo, hi)]
        else:
            if sel:
                out = out[out[col].astype(str).isin(sel)]
    return out

# -------------------------
# Detect columns based on your cluster CSV
# -------------------------
cluster_col = pick_col(df, ["cluster"])
gender_col = pick_col(df, ["gender"])
age_col = pick_col(df, ["age"])
cat_col = pick_col(df, ["category_label"])
mall_col = pick_col(df, ["shopping_mall_label"])
pay_col = pick_col(df, ["payment_method"])
qty_col = pick_col(df, ["quantity"])
price_orig_col = pick_col(df, ["price_original"])
price_col = pick_col(df, ["price"])
date_col = pick_col(df, ["invoice_date"])

# -------------------------
# Validate minimal requirements for spend
# -------------------------
missing_core = []
if cluster_col is None:
    missing_core.append("`cluster`")
if qty_col is None:
    missing_core.append("`quantity`")
if price_orig_col is None and price_col is None:
    missing_core.append("`price_original` atau `price`")

if missing_core:
    st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing_core))
    st.stop()

# Coerce numeric
df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
if price_orig_col is not None:
    df[price_orig_col] = pd.to_numeric(df[price_orig_col], errors="coerce")
if price_col is not None:
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

# Create total_spend for this page
spend_col = "total_spend_calc"
base_price = price_orig_col if price_orig_col is not None else price_col
df[spend_col] = pd.to_numeric(df[qty_col], errors="coerce") * pd.to_numeric(df[base_price], errors="coerce")

# Parse date (optional)
if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Normalize cluster type
df[cluster_col] = df[cluster_col].astype(str)

# -------------------------
# Sidebar-like controls on right (mirip Page 1)
# -------------------------
left, right = st.columns([3, 1], gap="large")

with right:
    st.markdown("### Filter Parameter")

    # Exclude: age + invoice_date (di cluster page ini date tidak dipakai untuk kontrol)
    excluded_controls = {age_col, date_col}
    excluded_controls = {c for c in excluded_controls if c is not None}

    preferred_controls = [
        cluster_col,
        gender_col,
        cat_col,
        mall_col,
        pay_col,
        qty_col,
        base_price,        # price_original atau price
    ]

    controls = []
    for c in preferred_controls:
        if c is None:
            continue
        if c in excluded_controls:
            continue
        if c in df.columns and c not in controls:
            controls.append(c)

    # Filter widgets
    filter_state = {}
    for col in controls:
        if col == cluster_col:
            opts = sorted(df[col].dropna().astype(str).unique().tolist())
            filter_state[col] = st.multiselect("cluster", options=opts, default=opts, key="c_cluster")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.dropna().empty:
                continue
            col_min = float(s.min())
            col_max = float(s.max())
            filter_state[col] = st.slider(col, col_min, col_max, (col_min, col_max), key=f"c_{col}")
        else:
            opts = sorted(df[col].dropna().astype(str).unique().tolist())
            filter_state[col] = st.multiselect(col, options=opts, default=opts, key=f"c_{col}")

    st.markdown("---")
    st.markdown("### Pengaturan Visual")

    group_by_options = [c for c in [cluster_col, gender_col, cat_col, mall_col, pay_col] if c is not None and c in df.columns]
    group_by = st.selectbox("Group by", options=group_by_options, index=0, key="c_groupby")

    sort_metric = st.radio("Sort by", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="c_sort")
    top_mode = st.radio("Tampilkan", ["Top N", "All"], horizontal=True, key="c_topmode")
    top_n = st.slider("Top N", 5, 30, 10, disabled=(top_mode == "All"), key="c_topn")
    pie_metric = st.radio("Pie berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="c_piemetric")

# Apply filters
df_f = apply_filters(df, filter_state)

with left:
    st.markdown('<div class="cluster-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="cluster-title">Ringkasan Data (Setelah Filter)</div>', unsafe_allow_html=True)

    total_rows = len(df_f)
    total_spend = float(pd.to_numeric(df_f[spend_col], errors="coerce").sum()) if total_rows else 0.0
    avg_spend = float(pd.to_numeric(df_f[spend_col], errors="coerce").mean()) if total_rows else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_cluster("Jumlah Transaksi", fmt_int(total_rows))
    with k2:
        render_kpi_cluster("Total Spend", fmt_money(total_spend), f"Sumber: {base_price} × {qty_col}")
    with k3:
        render_kpi_cluster("Rata-rata Spend", fmt_money(avg_spend))

    st.markdown('<div class="cluster-note">Hover pada chart untuk melihat detail.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if total_rows == 0:
        st.warning("Data kosong setelah filter.")
        st.stop()

    # -------------------------
    # Insight builder
    # -------------------------
    def insight_by(df_in: pd.DataFrame, group_col: str):
        out = (
            df_in.groupby(group_col, dropna=False)
            .agg(
                transaksi_count=(spend_col, "size"),
                total_spend_sum=(spend_col, "sum"),
                total_spend_avg=(spend_col, "mean"),
            )
            .reset_index()
        )
        return out

    insight = insight_by(df_f, group_by)

    sort_col = "total_spend_sum" if sort_metric == "Total Spend" else "transaksi_count"
    insight = insight.sort_values(sort_col, ascending=False)

    if top_mode == "Top N":
        insight = insight.head(top_n)

    rot = smart_xtick_rotation(insight[group_by].tolist())

    st.subheader("Visualisasi (Bar & Pie)")

    fig1 = px.bar(
        insight,
        x=group_by,
        y="total_spend_sum",
        hover_data=["transaksi_count", "total_spend_avg"],
        title="Total Spend (hasil kalkulasi)"
    )
    fig1.update_xaxes(tickangle=rot)
    st.plotly_chart(fig1, use_container_width=True, key="cluster_bar_spend")

    fig2 = px.bar(
        insight,
        x=group_by,
        y="transaksi_count",
        hover_data=["total_spend_sum", "total_spend_avg"],
        title="Jumlah Transaksi"
    )
    fig2.update_xaxes(tickangle=rot)
    st.plotly_chart(fig2, use_container_width=True, key="cluster_bar_trx")

    pie_value_col = "total_spend_sum" if pie_metric == "Total Spend" else "transaksi_count"
    pie_df = insight.copy()
    pie_df[pie_value_col] = pd.to_numeric(pie_df[pie_value_col], errors="coerce").fillna(0)
    pie_df = pie_df[pie_df[pie_value_col] > 0]

    if pie_df.empty:
        st.info("Pie chart tidak dapat ditampilkan (nilai 0/NaN semua).")
    else:
        fig3 = px.pie(
            pie_df,
            names=group_by,
            values=pie_value_col,
            hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
            title=f"Share {pie_metric}"
        )
        st.plotly_chart(fig3, use_container_width=True, key="cluster_pie")

    # -------------------------
    # Cluster profiling (otomatis, sinkron kolom CSV cluster)
    # -------------------------
    st.subheader("Profil Cluster (Ringkasan Otomatis)")

    def top_dist(df_in, col, topk=6):
        if col is None or col not in df_in.columns:
            return "-"
        vc = df_in[col].astype(str).value_counts(dropna=False)
        if vc.empty:
            return "-"
        top = vc.head(topk)
        total = vc.sum()
        parts = [f"{idx}: {cnt/total*100:,.2f}%" for idx, cnt in top.items()]
        return "<br/>".join(parts)

    clusters = sorted(df_f[cluster_col].dropna().astype(str).unique().tolist(), key=lambda x: (len(x), x))

    rows = []
    for c in clusters:
        d = df_f[df_f[cluster_col].astype(str) == str(c)].copy()
        size = len(d)
        size_pct = (size / len(df_f)) * 100 if len(df_f) else 0

        mean_price = float(pd.to_numeric(d[base_price], errors="coerce").mean()) if size else np.nan
        mean_spend = float(pd.to_numeric(d[spend_col], errors="coerce").mean()) if size else np.nan

        rows.append({
            "Cluster": c,
            "Size": size,
            "Size (%)": f"{size_pct:,.2f}%",
            "Mean Price": "-" if np.isnan(mean_price) else f"{mean_price:,.2f}",
            "Mean Spend": "-" if np.isnan(mean_spend) else f"{mean_spend:,.2f}",
            "Top Category": top_dist(d, cat_col, topk=6),
            "Top Mall": top_dist(d, mall_col, topk=6),
        })

    prof = pd.DataFrame(rows)

    def df_to_html_table(prof_df: pd.DataFrame) -> str:
        cols = prof_df.columns.tolist()
        thead = "<thead><tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr></thead>"

        body_rows = []
        for _, r in prof_df.iterrows():
            tds = []
            for c in cols:
                val = r[c]
                if c in ["Cluster", "Size", "Size (%)", "Mean Price", "Mean Spend"]:
                    tds.append(f'<td class="ct-col">{val}</td>')
                elif c in ["Top Category", "Top Mall"]:
                    tds.append(f'<td class="ct-value">{val}</td>')
                else:
                    tds.append(f"<td>{val}</td>")
            body_rows.append("<tr>" + "".join(tds) + "</tr>")
        tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

        return f'<div class="cluster-box"><table class="cluster-table">{thead}{tbody}</table></div>'

    st.markdown(df_to_html_table(prof), unsafe_allow_html=True)

    with st.expander("Preview data (hasil filter)"):
        st.dataframe(df_f.head(50), use_container_width=True, height=360)

    with st.expander("Download data (hasil filter)"):
        csv_bytes = df_f.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV (filtered)",
            data=csv_bytes,
            file_name="cluster_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
