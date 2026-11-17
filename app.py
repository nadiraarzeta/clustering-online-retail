
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Memuat model dan scaler
package = joblib.load("deployment_model/clustering_deployment.joblib")

model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

# Keterangan cluster (dari profil RFM di Praktikum 13)
keterangan = {
    0: "Pelanggan cukup aktif, frekuensi dan nilai pembelian rendah-menengah. Rekomendasi: tawarkan produk sesuai riwayat pembelian.",
    1: "Pelanggan sudah lama tidak bertransaksi dan pembeliannya rendah. Rekomendasi: kampanye reaktivasi (voucher, pengingat).",
    2: "Pelanggan aktif, sering membeli, nilai pembelian tinggi. Rekomendasi: program loyalitas dan penawaran eksklusif.",
}

st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means berdasarkan RFM.")

recency = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0, value=15)
frequency = st.number_input("Frequency (jumlah transaksi)", min_value=1, value=8)
monetary = st.number_input("Monetary (total belanja, £)", min_value=1.0, value=2500.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)],
    })

    data_scaled = scaler.transform(data[feature_columns])
    cluster = int(model.predict(data_scaled)[0])

    st.success(f"Pelanggan masuk ke Cluster {cluster}")
    st.info(keterangan[cluster])
