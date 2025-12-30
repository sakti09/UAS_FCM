import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Mall Insight", layout="wide")

# keterangan: load CSS utama untuk seluruh halaman
def load_css(path: str = "assets/insight.css"):
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

load_css()

# keterangan: judul dan deskripsi halaman utama
st.title("Mall Insight Dashboard")
st.caption("Halaman utama dashboard analisis. Navigasi halaman tersedia pada sidebar kiri.")

# keterangan: informasi halaman Insight (Page 1)
st.markdown(
    """
    <div class="landing-box">
        <div style="font-weight:900; font-size:16px; margin-bottom:6px; color:#1F3020;">
            Page 1 — Insight
        </div>
        <div style="color:rgba(31,48,32,0.85); line-height:1.55;">
            Halaman ini menyajikan analisis eksploratif terhadap data transaksi,
            yang mencakup:
            <ul style="margin:8px 0 0 18px;">
              <li>Peninjauan dataset (View Dataset)</li>
              <li>Analisis berdasarkan parameter</li>
              <li>Tren tahunan</li>
              <li>Tren bulanan</li>
            </ul>
            Analisis digunakan untuk memahami pola transaksi dan perilaku pembelian secara umum.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# keterangan: informasi halaman Clustering (Page 2)
st.markdown(
    """
    <div class="landing-box">
        <div style="font-weight:900; font-size:16px; margin-bottom:6px; color:#1F3020;">
            Page 2 — Clustering
        </div>
        <div style="color:rgba(31,48,32,0.85); line-height:1.55;">
            Halaman ini menampilkan hasil segmentasi pelanggan berbasis clustering,
            meliputi:
            <ul style="margin:8px 0 0 18px;">
              <li>Distribusi jumlah dan proporsi cluster</li>
              <li>Rata-rata harga transaksi per cluster</li>
              <li>Distribusi kategori produk per cluster</li>
              <li>Distribusi lokasi mall per cluster</li>
            </ul>
            Segmentasi digunakan untuk mengidentifikasi karakteristik utama setiap kelompok pelanggan.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.caption("Gunakan sidebar kiri untuk berpindah ke halaman Insight atau Clustering.")
