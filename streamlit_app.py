import streamlit as st
import pandas as pd
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FUNGSI UNTUK MEMUAT SEMUA ARTEFAK ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('obesity_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        model_columns = joblib.load('model_columns.pkl') # <-- MEMUAT NAMA KOLOM
        return model, scaler, label_encoder, model_columns
    except FileNotFoundError as e:
        st.error(f"Error memuat file: {e}. Pastikan semua file .pkl (model, scaler, encoder, dan columns) ada di folder yang sama.")
        return None, None, None, None

model, scaler, label_encoder, model_columns = load_artifacts()

# --- 3. FUNGSI UNTUK PREDIKSI ---
def predict_obesity(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# --- 4. USER INTERFACE (UI) ---
st.title("Aplikasi Prediksi Tingkat Obesitas")
st.write("Aplikasi ini menggunakan model machine learning untuk memprediksi tingkat obesitas.")

if model is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Data Diri & Keluarga")
        age = st.number_input("Umur (Age)", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
        height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=220.0, value=170.0) / 100.0
        weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0)
        family_history = st.radio("Riwayat Obesitas Keluarga", ["yes", "no"])

    with col2:
        st.header("Kebiasaan Makan & Aktivitas")
        favc = st.radio("Sering Konsumsi Makanan Tinggi Kalori (FAVC)?", ["yes", "no"])
        fcvc = st.slider("Frekuensi Konsumsi Sayur (FCVC)", 1, 3, 2, step=1)
        ncp = st.slider("Jumlah Makan Utama Sehari (NCP)", 1, 4, 3, step=1)
        caec = st.selectbox("Konsumsi Cemilan (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Konsumsi Alkohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Konsumsi Air (liter/hari) (CH2O)", 1, 3, 2, step=1)
        faf = st.slider("Frekuensi Aktivitas Fisik (hari/minggu) (FAF)", 0, 3, 1, step=1)
        tue = st.slider("Waktu Menggunakan Gadget (jam/hari) (TUE)", 0, 2, 1, step=1)

    st.header("Kebiasaan Lain")
    smoke = st.radio("Apakah Anda Merokok (SMOKE)?", ["yes", "no"])
    scc = st.radio("Apakah Anda Memantau Kalori (SCC)?", ["yes", "no"])
    mtrans = st.selectbox("Transportasi Utama (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    if st.button("Prediksi Tingkat Obesitas", type="primary"):
        # --- 5. PROSES DATA UNTUK PREDIKSI ---
        
        # Buat dictionary untuk input. Nama kolom diambil dari file yang sudah pasti benar.
        input_dict = {col: 0 for col in model_columns}

        # Mengisi nilai numerik
        input_dict['Age'] = age
        input_dict['Height'] = height
        input_dict['Weight'] = weight
        input_dict['FCVC'] = fcvc
        input_dict['NCP'] = ncp
        input_dict['CH2O'] = ch2o
        input_dict['FAF'] = faf
        input_dict['TUE'] = tue

        # Mengisi nilai untuk kolom dummy (diasumsikan DILATIH DENGAN drop_first=False)
        input_dict[f'Gender_{gender}'] = 1
        input_dict[f'family_history_with_overweight_{family_history}'] = 1
        input_dict[f'FAVC_{favc}'] = 1
        input_dict[f'SCC_{scc}'] = 1
        input_dict[f'SMOKE_{smoke}'] = 1
        input_dict[f'CAEC_{caec}'] = 1
        input_dict[f'CALC_{calc}'] = 1
        input_dict[f'MTRANS_{mtrans}'] = 1
        
        # Konversi ke DataFrame dan pastikan urutan kolom benar
        input_df = pd.DataFrame([input_dict])[model_columns]
        
        # Lakukan prediksi
        try:
            hasil_prediksi = predict_obesity(input_df)
            st.success(f"Hasil Prediksi: Anda kemungkinan termasuk dalam kategori **{hasil_prediksi}**")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
