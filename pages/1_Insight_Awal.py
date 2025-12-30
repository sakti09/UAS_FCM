import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Insight", layout="wide")

# LOAD CSS FROM FILE
def load_css(path: str = "assets/insight.css"):
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file tidak ditemukan: {path}")

load_css()

st.title("Page 1 — Insight Dashboard")

#LOAD DATA
uploaded = st.file_uploader("Upload CSV (final insight)", type=["csv"])
if not uploaded:
    st.info("Upload CSV dulu untuk melihat dashboard insight.")
    st.stop()

df = pd.read_csv(uploaded)

# ROUTER
if "insight_subpage" not in st.session_state:
    st.session_state.insight_subpage = "home"

def go(page_name: str):
    st.session_state.insight_subpage = page_name

def card_button(label: str, key: str, on_click_page: str, color_class="card-1", disabled=False):
    btn_label = f"{label}\n\nKlik untuk buka"
    st.markdown(f'<div class="card {color_class}">', unsafe_allow_html=True)
    if st.button(btn_label, key=key, disabled=disabled, use_container_width=True):
        go(on_click_page)
    st.markdown("</div>", unsafe_allow_html=True)

# HELPERS
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

def render_kpi(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="kpi-wrap">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def insight_by(df_in: pd.DataFrame, group_col: str):
    return (
        df_in.groupby(group_col, dropna=False)
        .agg(
            transaksi_count=("total_spend", "size"),
            total_spend_sum=("total_spend", "sum"),
            total_spend_avg=("total_spend", "mean"),
        )
        .reset_index()
    )

def smart_xtick_rotation(values) -> int:
    vals = [str(v) for v in values]
    if not vals:
        return 0
    maxlen = max(len(v) for v in vals)
    n = len(vals)
    return 45 if (maxlen >= 12 or n >= 8) else 0

def apply_filters(df_in: pd.DataFrame, filter_state: dict) -> pd.DataFrame:
    df_f = df_in.copy()
    for col, sel in filter_state.items():
        if pd.api.types.is_numeric_dtype(df_f[col]):
            lo, hi = sel
            df_f = df_f[pd.to_numeric(df_f[col], errors="coerce").between(lo, hi)]
        else:
            if sel:
                df_f = df_f[df_f[col].astype(str).isin(sel)]
    return df_f

def year_theme(year: int):
    if year == 2021:
        panel_bg = "background: linear-gradient(135deg, rgba(255,105,180,0.16), rgba(255,0,80,0.10));"
        colors = ["#FF4D8D", "#FF7AAE", "#FF9BC6", "#FF2D6A", "#FFC0D9"]
        return panel_bg, colors, colors
    if year == 2022:
        panel_bg = "background: linear-gradient(135deg, rgba(255,200,0,0.18), rgba(255,120,0,0.10));"
        colors = ["#FFC400", "#FFB703", "#FB8500", "#FFD166", "#FFE08A"]
        return panel_bg, colors, colors
    panel_bg = "background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(0,180,180,0.10));"
    colors = ["#22C55E", "#10B981", "#34D399", "#06B6D4", "#A7F3D0"]
    return panel_bg, colors, colors

# HOME (MENU)
if st.session_state.insight_subpage == "home":
    st.subheader("Menu (Page 1)")
    st.caption("Klik kartu di bawah untuk membuka sub-halaman (masih di Page 1).")

    cols = st.columns(5)
    with cols[0]:
        card_button("1) View Dataset", "card_1", "view_dataset", color_class="card-1")
    with cols[1]:
        card_button("2) Insight by Parameter", "card_2", "insight_param", color_class="card-2")
    with cols[2]:
        card_button("3) Tren Tahunan", "card_3", "trend_yearly", color_class="card-3")
    with cols[3]:
        card_button("4) Tren Bulanan", "card_4", "trend_monthly", color_class="card-4")
    with cols[4]:
        card_button("5) Placeholder", "card_5", "todo_5", color_class="card-5", disabled=True)

    st.markdown("---")
    st.caption("Preview data:")
    st.dataframe(df.head(5), use_container_width=True)

# SUBPAGE: VIEW DATASET (MENU 1)
elif st.session_state.insight_subpage == "view_dataset":
    topbar = st.columns([1, 6])
    with topbar[0]:
        if st.button("⬅️ Back", use_container_width=True):
            go("home")
    with topbar[1]:
        st.subheader("View Dataset")

    st.caption("Tabel data lengkap yang dapat diurutkan (sorting).")

    sort_cols = df.columns.tolist()
    c1, c2, c3 = st.columns([2, 1, 2])

    with c1:
        default_idx = sort_cols.index("age") if "age" in sort_cols else 0
        sort_by = st.selectbox("Sort by column", options=sort_cols, index=default_idx)
    with c2:
        ascending = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
    with c3:
        n_rows = st.slider("Jumlah baris ditampilkan", 10, 500, 100)

    df_sorted = df.sort_values(by=sort_by, ascending=(ascending == "Ascending"))
    st.markdown("---")
    st.dataframe(df_sorted.head(n_rows), use_container_width=True, height=560)

# SUBPAGE: INSIGHT PARAM (MENU 2)
elif st.session_state.insight_subpage == "insight_param":
    topbar = st.columns([1, 6])
    with topbar[0]:
        if st.button("⬅️ Back", use_container_width=True):
            go("home")
    with topbar[1]:
        st.subheader("Insight by Parameter")

    if "total_spend" not in df.columns:
        st.error("Kolom `total_spend` tidak ditemukan.")
        st.stop()

    excluded_controls = {"age", "invoice_date_time", "invoice_date_day", "invoice_date_month", "invoice_date_year"}
    controls = ["gender", "category", "quantity", "payment_method", "shopping_mall", "age_class", "price_class", "price"]
    controls = [c for c in controls if c in df.columns and c not in excluded_controls]

    left, right = st.columns([3, 1])

    with right:
        st.markdown("### Filter Parameter")
        filter_state = {}
        for col in controls:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_min = float(pd.to_numeric(df[col], errors="coerce").min())
                col_max = float(pd.to_numeric(df[col], errors="coerce").max())
                filter_state[col] = st.slider(col, col_min, col_max, (col_min, col_max), key=f"p_{col}")
            else:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                filter_state[col] = st.multiselect(col, options=opts, default=opts, key=f"p_{col}")

        st.markdown("---")
        group_by_options = [c for c in ["gender", "category", "payment_method", "shopping_mall", "age_class", "price_class", "quantity", "price"] if c in df.columns]
        group_by = st.selectbox("Group by", options=group_by_options, index=0)

        sort_metric = st.radio("Sort by", ["Total Spend", "Jumlah Transaksi"], horizontal=True)
        top_mode = st.radio("Tampilkan", ["Top N", "All"], horizontal=True)
        top_n = st.slider("Top N", 5, 30, 10, disabled=(top_mode == "All"))
        pie_metric = st.radio("Pie berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True)

    df_f = apply_filters(df, filter_state)

    with left:
        total_trx = len(df_f)
        total_spend = float(pd.to_numeric(df_f["total_spend"], errors="coerce").sum())
        avg_spend = float(pd.to_numeric(df_f["total_spend"], errors="coerce").mean()) if total_trx > 0 else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi("Jumlah Transaksi", fmt_int(total_trx))
        with k2:
            render_kpi("Total Spend", fmt_money(total_spend))
        with k3:
            render_kpi("Rata-rata Spend", fmt_money(avg_spend))

        st.markdown("---")
        if total_trx == 0:
            st.warning("Data kosong setelah filter.")
        else:
            insight = insight_by(df_f, group_by)
            insight = insight.sort_values("transaksi_count" if sort_metric == "Jumlah Transaksi" else "total_spend_sum", ascending=False)
            if top_mode == "Top N":
                insight = insight.head(top_n)

            rot = smart_xtick_rotation(insight[group_by].tolist())

            fig1 = px.bar(insight, x=group_by, y="total_spend_sum",
                          hover_data=["transaksi_count", "total_spend_avg"], title="Total Spend")
            fig1.update_xaxes(tickangle=rot)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.bar(insight, x=group_by, y="transaksi_count",
                          hover_data=["total_spend_sum", "total_spend_avg"], title="Jumlah Transaksi")
            fig2.update_xaxes(tickangle=rot)
            st.plotly_chart(fig2, use_container_width=True)

            pie_value_col = "total_spend_sum" if pie_metric == "Total Spend" else "transaksi_count"
            fig3 = px.pie(insight, names=group_by, values=pie_value_col,
                          hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
                          title=f"Share {pie_metric}")
            st.plotly_chart(fig3, use_container_width=True)

# SUBPAGE: TREND YEARLY (MENU 3)
elif st.session_state.insight_subpage == "trend_yearly":
    topbar = st.columns([1, 6])
    with topbar[0]:
        if st.button("⬅️ Back", use_container_width=True):
            go("home")
    with topbar[1]:
        st.subheader("Tren Tahunan (2021 | 2022 | 2023)")
        st.caption("3 kolom: KPI → Bar → Pie. Bar bisa dipilih Spend / Transaksi.")

    if "invoice_date_year" not in df.columns:
        st.error("Kolom `invoice_date_year` tidak ditemukan.")
        st.stop()
    if "total_spend" not in df.columns:
        st.error("Kolom `total_spend` tidak ditemukan.")
        st.stop()

    excluded_controls = {"age", "invoice_date_time", "invoice_date_day", "invoice_date_month", "invoice_date_year"}
    controls = ["gender", "category", "quantity", "payment_method", "shopping_mall", "age_class", "price_class", "price"]
    controls = [c for c in controls if c in df.columns and c not in excluded_controls]

    main_left, main_right = st.columns([3, 1])

    with main_right:
        st.markdown("### Kontrol (Global)")
        filter_state = {}
        for col in controls:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_min = float(pd.to_numeric(df[col], errors="coerce").min())
                col_max = float(pd.to_numeric(df[col], errors="coerce").max())
                filter_state[col] = st.slider(col, col_min, col_max, (col_min, col_max), key=f"y_{col}")
            else:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                filter_state[col] = st.multiselect(col, options=opts, default=opts, key=f"y_{col}")

        st.markdown("---")
        group_by_options = [c for c in ["gender", "category", "payment_method", "shopping_mall", "age_class", "price_class", "quantity", "price"] if c in df.columns]
        group_by = st.selectbox("Group by", options=group_by_options, index=0, key="y_group")

        sort_metric = st.radio("Sort by", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="y_sort_metric")
        bar_metric = st.radio("Bar berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="y_bar_metric")

        top_mode = st.radio("Tampilkan", ["Top N", "All"], horizontal=True, key="y_top_mode")
        top_n = st.slider("Top N", 5, 25, 10, disabled=(top_mode == "All"), key="y_top_n")

        pie_metric = st.radio("Pie berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="y_pie_metric")

    df_global = apply_filters(df, filter_state)

    def panel_year(container, year: int):
        df_y = df_global[df_global["invoice_date_year"] == year].copy()
        panel_bg, bar_colors, pie_colors = year_theme(year)

        with container:
            st.markdown(f'<div class="year-panel" style="{panel_bg}">', unsafe_allow_html=True)
            st.markdown(f'<div class="year-title">Tahun {year}</div>', unsafe_allow_html=True)

            total_trx = len(df_y)
            total_spend = float(pd.to_numeric(df_y["total_spend"], errors="coerce").sum())
            avg_spend = float(pd.to_numeric(df_y["total_spend"], errors="coerce").mean()) if total_trx > 0 else 0.0

            k1, k2, k3 = st.columns(3)
            with k1:
                render_kpi("Transaksi", fmt_int(total_trx))
            with k2:
                render_kpi("Total Spend", fmt_money(total_spend))
            with k3:
                render_kpi("Avg Spend", fmt_money(avg_spend))

            if total_trx == 0:
                st.caption("Data kosong.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            insight = insight_by(df_y, group_by)
            sort_col = "total_spend_sum" if sort_metric == "Total Spend" else "transaksi_count"
            insight = insight.sort_values(sort_col, ascending=False)

            if top_mode == "Top N":
                insight = insight.head(top_n)

            rot = smart_xtick_rotation(insight[group_by].tolist())

            y_col = "total_spend_sum" if bar_metric == "Total Spend" else "transaksi_count"
            bar_title = "Total Spend" if bar_metric == "Total Spend" else "Jumlah Transaksi"

            fig_bar = px.bar(
                insight,
                x=group_by,
                y=y_col,
                hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
                title=bar_title,
                color_discrete_sequence=bar_colors
            )
            fig_bar.update_xaxes(tickangle=rot)
            fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True, key=f"y_{year}_bar_{group_by}_{y_col}_{top_mode}_{top_n}")

            pie_val = "total_spend_sum" if pie_metric == "Total Spend" else "transaksi_count"
            fig_pie = px.pie(
                insight,
                names=group_by,
                values=pie_val,
                hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
                title=f"Share {pie_metric}",
                color_discrete_sequence=pie_colors
            )
            fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True, key=f"y_{year}_pie_{group_by}_{pie_val}_{top_mode}_{top_n}")

            st.markdown("</div>", unsafe_allow_html=True)

    with main_left:
        c2021, c2022, c2023 = st.columns(3, gap="medium")
        panel_year(c2021, 2021)
        panel_year(c2022, 2022)
        panel_year(c2023, 2023)

# SUBPAGE: TREND MONTHLY (MENU 4)
elif st.session_state.insight_subpage == "trend_monthly":
    topbar = st.columns([1, 6])
    with topbar[0]:
        if st.button("⬅️ Back", use_container_width=True):
            go("home")
    with topbar[1]:
        st.subheader("Tren Bulanan (Jan–Dec)")
        st.caption("Atas: 12 mini bar chart (tanpa scroll). Bawah: pie chart per bulan (scroll).")

    required_cols = ["invoice_date_year", "invoice_date_month", "total_spend"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {missing}")
        st.stop()

    excluded_controls = {"age", "invoice_date_time", "invoice_date_day", "invoice_date_month", "invoice_date_year"}
    controls = ["gender", "category", "quantity", "payment_method", "shopping_mall", "age_class", "price_class", "price"]
    controls = [c for c in controls if c in df.columns and c not in excluded_controls]

    main_left, main_right = st.columns([3, 1])

    with main_right:
        st.markdown("### Kontrol (Global)")
        year_pick = st.selectbox("Filter Tahun", ["All", "2021", "2022", "2023"], index=0, key="m_year_pick")

        filter_state = {}
        for col in controls:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = pd.to_numeric(df[col], errors="coerce")
                col_min = float(s.min())
                col_max = float(s.max())
                filter_state[col] = st.slider(col, col_min, col_max, (col_min, col_max), key=f"m_{col}")
            else:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                filter_state[col] = st.multiselect(col, options=opts, default=opts, key=f"m_{col}")

        st.markdown("---")
        group_by_options = [c for c in ["gender", "category", "payment_method", "shopping_mall", "age_class", "price_class", "quantity", "price"] if c in df.columns]
        group_by = st.selectbox("Group by", options=group_by_options, index=0, key="m_group")

        sort_metric = st.radio("Sort by", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="m_sort")
        bar_metric = st.radio("Bar berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="m_bar")

        top_mode = st.radio("Tampilkan", ["Top N", "All"], horizontal=True, key="m_top")
        top_n = st.slider("Top N", 3, 25, 10, disabled=(top_mode == "All"), key="m_topn")

        pie_metric = st.radio("Pie berdasarkan", ["Total Spend", "Jumlah Transaksi"], horizontal=True, key="m_pie")

    df_global = apply_filters(df, filter_state).copy()

    if year_pick != "All":
        df_global = df_global[df_global["invoice_date_year"] == int(year_pick)].copy()

    df_global["invoice_date_month"] = pd.to_numeric(df_global["invoice_date_month"], errors="coerce")

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    colA = [1, 4, 7, 10]
    colB = [2, 5, 8, 11]
    colC = [3, 6, 9, 12]

    def month_panel(container, m: int, show_pie: bool, section_tag: str):
        df_m = df_global[df_global["invoice_date_month"] == m].copy()

        with container:
            st.markdown('<div class="month-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="month-title">{month_names[m]} (Month {m})</div>', unsafe_allow_html=True)

            if df_m.empty:
                st.caption("Data kosong.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            insight = insight_by(df_m, group_by)

            insight["total_spend_sum"] = pd.to_numeric(insight["total_spend_sum"], errors="coerce")
            insight["transaksi_count"] = pd.to_numeric(insight["transaksi_count"], errors="coerce")
            insight["total_spend_avg"] = pd.to_numeric(insight["total_spend_avg"], errors="coerce")

            sort_col = "total_spend_sum" if sort_metric == "Total Spend" else "transaksi_count"
            insight = insight.dropna(subset=[group_by, sort_col]).sort_values(sort_col, ascending=False)

            if top_mode == "Top N":
                insight = insight.head(top_n)

            if insight.empty:
                st.caption("Insight kosong setelah cleaning.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            rot = smart_xtick_rotation(insight[group_by].tolist())

            # BAR
            y_col = "total_spend_sum" if bar_metric == "Total Spend" else "transaksi_count"
            title_bar = "Total Spend" if bar_metric == "Total Spend" else "Jumlah Transaksi"

            fig_bar = px.bar(
                insight,
                x=group_by,
                y=y_col,
                hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
                title=title_bar
            )
            fig_bar.update_xaxes(tickangle=rot)
            fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))

            # ✅ KEY UNIK: bedakan antara section bar-only vs section pie
            bar_key = f"m_{section_tag}_bar_month{m}_{group_by}_{y_col}_{year_pick}_{top_mode}_{top_n}"
            st.plotly_chart(fig_bar, use_container_width=True, key=bar_key)

            # PIE
            if show_pie:
                pie_col = "total_spend_sum" if pie_metric == "Total Spend" else "transaksi_count"
                pie_df = insight.dropna(subset=[group_by, pie_col]).copy()
                pie_df[pie_col] = pd.to_numeric(pie_df[pie_col], errors="coerce")
                pie_df = pie_df[pie_df[pie_col] > 0]

                if pie_df.empty:
                    st.caption("Pie chart tidak bisa ditampilkan (nilai 0/NaN).")
                else:
                    fig_pie = px.pie(
                        pie_df,
                        names=group_by,
                        values=pie_col,
                        hover_data=["total_spend_sum", "transaksi_count", "total_spend_avg"],
                        title=f"Share {pie_metric}"
                    )
                    fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))

                    pie_key = f"m_{section_tag}_pie_month{m}_{group_by}_{pie_col}_{year_pick}_{top_mode}_{top_n}"
                    st.plotly_chart(fig_pie, use_container_width=True, key=pie_key)

            st.markdown("</div>", unsafe_allow_html=True)

    with main_left:
        st.markdown("### Mini Bar Chart per Bulan (tanpa scroll)")
        a, b, c = st.columns(3, gap="medium")
        for m in colA:
            month_panel(a, m, show_pie=False, section_tag="top")
        for m in colB:
            month_panel(b, m, show_pie=False, section_tag="top")
        for m in colC:
            month_panel(c, m, show_pie=False, section_tag="top")

        st.markdown("---")
        st.markdown("### Pie Chart per Bulan (scroll)")
        a2, b2, c2 = st.columns(3, gap="medium")
        for m in colA:
            month_panel(a2, m, show_pie=True, section_tag="bottom")
        for m in colB:
            month_panel(b2, m, show_pie=True, section_tag="bottom")
        for m in colC:
            month_panel(c2, m, show_pie=True, section_tag="bottom")

else:
    go("home")
