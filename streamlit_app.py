# app.py (Input di Halaman Utama dengan Kolom)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI HALAMAN ---
# Kita buat layout "wide" agar lebih pas untuk tampilan kolom
st.set_page_config(
    page_title="Dashboard Prediksi Obesitas",
    page_icon="⚖️",
    layout="wide", # Mengubah layout menjadi lebar
    initial_sidebar_state="collapsed" # Sidebar bisa kita sembunyikan
)

# --- FUNGSI UNTUK MEMUAT SEMUA ASET ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        cat_encoders = joblib.load("categorical_label_encoders.pkl")
        target_encoder = joblib.load("target_label_encoder.pkl")
        return model, scaler, cat_encoders, target_encoder
    except FileNotFoundError as e:
        st.error(f"Error: File aset tidak ditemukan. Pastikan semua file .pkl ada di folder yang sama dengan app.py.")
        st.error(e)
        return None, None, None, None

# Memuat semua aset
model, scaler, cat_encoders, target_encoder = load_assets()


# --- JUDUL DAN DESKRIPSI DASHBOARD ---
st.title("⚖️ Dashboard Prediksi Tingkat Obesitas")
st.write(
    "Dashboard ini menggunakan model Random Forest untuk memprediksi tingkat obesitas "
    "berdasarkan beberapa fitur gaya hidup dan atribut fisik. Silakan masukkan data Anda di bawah ini."
)
st.write("---")


# --- FUNGSI INPUT PENGGUNA DI HALAMAN UTAMA ---
# Tidak lagi menggunakan st.sidebar
def user_inputs():
    # Membuat dua kolom utama untuk input
    col1, col2 = st.columns(2, gap="large")

    # Input di Kolom 1
    with col1:
        st.subheader("Data Diri & Fisik")
        age = st.slider("Usia (Tahun)", 1, 100, 25)
        gender = st.selectbox("Jenis Kelamin (Gender)", ("Male", "Female"))
        height = st.slider("Tinggi Badan (m)", 1.0, 2.2, 1.70)
        weight = st.slider("Berat Badan (kg)", 30, 200, 70)
        
        st.subheader("Aktivitas & Kebiasaan Lain")
        smoke = st.selectbox("Apakah Anda merokok (SMOKE)?", ("yes", "no"))
        faf = st.slider("Frekuensi aktivitas fisik per minggu (FAF)", 0.0, 3.0, 1.0)
        tue = st.slider("Waktu penggunaan gawai per hari (Jam) (TUE)", 0.0, 2.0, 1.0)
        mtrans = st.selectbox("Moda transportasi utama (MTRANS)", 
                                ("Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"))

    # Input di Kolom 2
    with col2:
        st.subheader("Kebiasaan Makan & Gaya Hidup")
        family_history = st.selectbox("Riwayat obesitas keluarga (family_history_with_overweight)", ("yes", "no"))
        favc = st.selectbox("Sering makan makanan tinggi kalori (FAVC)", ("yes", "no"))
        scc = st.selectbox("Apakah Anda memonitor asupan kalori (SCC)?", ("yes", "no"))
        fcvc = st.slider("Frekuensi makan sayur (FCVC)", 1.0, 3.0, 2.0)
        ncp = st.slider("Jumlah makan utama per hari (NCP)", 1.0, 4.0, 3.0)
        caec = st.selectbox("Makan di antara waktu makan (CAEC)", ("no", "Sometimes", "Frequently", "Always"))
        ch2o = st.slider("Konsumsi air per hari (Liter) (CH2O)", 1.0, 4.0, 2.0)
        calc = st.selectbox("Konsumsi alkohol (CALC)", ("no", "Sometimes", "Frequently", "Always"))
        

    input_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history_with_overweight': family_history, 'FAVC': favc, 'FCVC': fcvc,
        'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf,
        'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
    }
    
    features = pd.DataFrame(input_data, index=[0])
    return features

# --- MULAI AREA INPUT ---
# Memastikan semua aset berhasil dimuat sebelum menampilkan input
if all(asset is not None for asset in [model, scaler, cat_encoders, target_encoder]):
    
    # Menampilkan header untuk area input
    st.header("📝 Masukkan Data Anda")

    input_df_unordered = user_inputs()

    # Mengatur urutan kolom agar sesuai dengan model
    try:
        correct_order = scaler.feature_names_in_
        input_df = input_df_unordered[correct_order]
    except Exception as e:
        st.error(f"Terjadi error saat menyusun urutan fitur: {e}")
        st.stop()
    
    st.write("---")

    # --- TOMBOL PREDIKSI DAN TAMPILAN HASIL ---
    # Menggunakan st.columns untuk menengahkan tombol
    col_button1, col_button2, col_button3 = st.columns([1, 1.5, 1])
    with col_button2:
        predict_button = st.button("Prediksi Tingkat Obesitas", type="primary", use_container_width=True)

    if predict_button:
        # --- PREPROCESSING ---
        processed_df = input_df.copy()

        for col, encoder in cat_encoders.items():
            if col in processed_df.columns:
                processed_df[col] = encoder.transform(processed_df[[col]])[0]
        
        original_columns = processed_df.columns
        processed_df_scaled = scaler.transform(processed_df)
        processed_df = pd.DataFrame(processed_df_scaled, columns=original_columns)

        # --- PREDIKSI ---
        prediction_numeric = model.predict(processed_df)
        prediction_categorical = target_encoder.inverse_transform(prediction_numeric)
        
        # --- TAMPILKAN HASIL ---
        st.subheader("Hasil Prediksi")
        
        kategori_prediksi = prediction_categorical[0]
        
        if "Obesity" in kategori_prediksi:
            st.error(f"**Kategori Prediksi: {kategori_prediksi}**", icon="🚨")
            st.warning("Berdasarkan data yang dimasukkan, Anda berisiko tinggi mengalami obesitas. Disarankan untuk berkonsultasi dengan profesional kesehatan.")
        elif "Overweight" in kategori_prediksi:
            st.warning(f"**Kategori Prediksi: {kategori_prediksi}**", icon="⚠️")
            st.info("Anda berada dalam kategori berat badan berlebih. Memperbaiki pola makan dan meningkatkan aktivitas fisik dapat membantu mencapai berat badan ideal.")
        elif "Normal_Weight" in kategori_prediksi:
            st.success(f"**Kategori Prediksi: {kategori_prediksi}**", icon="✅")
            st.info("Selamat! Berat badan Anda berada dalam kategori normal. Pertahankan gaya hidup sehat Anda.")
        else: # Insufficient_Weight
            st.info(f"**Kategori Prediksi: {kategori_prediksi}**", icon="ℹ️")
            st.info("Berat badan Anda di bawah normal. Pastikan asupan nutrisi Anda cukup untuk mendukung kesehatan tubuh.")

        st.write("")
        st.markdown("_**Disclaimer:** Hasil prediksi ini berdasarkan model dan tidak menggantikan diagnosis medis profesional._")
