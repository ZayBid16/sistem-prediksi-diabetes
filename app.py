import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =====================================================================
# KONFIGURASI HALAMAN UTAMA STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Sistem Deteksi Risiko Diabetes - Naive Bayes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# KUSTOMISASI DESAIN (CSS) UNTUK TAMPILAN DARK MODE PREMIUM
# =====================================================================
st.markdown("""
    <style>
    /* Mengatur warna latar belakang halaman utama (Gelap) */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Mempercantik tampilan kontainer/card agar kontras dengan background gelap */
    .custom-card {
        background-color: #1e293b;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    
    /* Judul Utama dengan warna cerah neon */
    .main-title {
        color: #38bdf8;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        font-size: 2.5rem;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    
    /* Subtitle dengan aksen hijau neon */
    .sub-title {
        color: #2dd4bf;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.2rem;
    }
    
    /* Desain Tombol Prediksi (Teal Neon) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.6);
    }

    /* Memaksa label input dan teks streamlit menjadi putih agar mudah dibaca */
    label, .css-81oif8, .css-1qgn86z {
        color: #f1f5f9 !important;
    }
    
    /* Desain Kotak Alert Kustom */
    .custom-alert-success {
        background-color: #064e3b;
        border-left: 5px solid #059669;
        padding: 15px;
        border-radius: 8px;
        color: #ecfdf5;
        margin-bottom: 15px;
    }

    .custom-alert-error {
        background-color: #7f1d1d;
        border-left: 5px solid #dc2626;
        padding: 15px;
        border-radius: 8px;
        color: #fef2f2;
        margin-bottom: 15px;
    }
    
    /* Pengaturan Sidebar Gelap */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Pengaturan tabel data dark mode */
    .stTable, table {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# PENGATURAN STRUKTUR DATA UTAMA & MODEL
# =====================================================================
@st.cache_resource
def load_diabetes_model():
    """Fungsi untuk memuat model PKL secara aman"""
    try:
        model = joblib.load('model_diabetes.pkl')
        return model, False 
    except FileNotFoundError:
        return None, True 

model, is_simulation = load_diabetes_model()

# =====================================================================
# SIDEBAR (MENU SAMPING) - DETAIL MAHASISWA & METODOLOGI
# =====================================================================
with st.sidebar:
    st.markdown("<div style='text-align: center;'><img src='https://img.icons8.com/clouds/150/hospital.png' width='120'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #38bdf8;'>🎓 PROYEK AKHIR</h3>", unsafe_allow_html=True)
    
    # Menampilkan informasi peneliti sesuai proposal asli
    st.info("""
    **Penyusun Kelompok:**
    * **Zaidan Muhammad Abid**
      *(NPM: 202310715260)*
    * **Muhammad Irfan Adji Wicaksono**
      *(NPM: 202310715153)*
    
    **Mata Kuliah:**
    Penambangan Data (Data Mining)
    
    **Institusi:**
    Informatika, Fakultas Ilmu Komputer
    **Universitas Bhayangkara Jakarta Raya**
    """)
    
    st.success("""
    * **Model:** Naive Bayes (Gaussian)
    * **Akurasi Model:** 75.32%
    * **Pembagian Data:** 80:20 Split
    * **Dataset:** Pima Indians Diabetes
    """)
    
    st.markdown("---")
    st.caption("© 2026 - Zaidan & Irfan (Informatika Ubharajaya)")

# =====================================================================
# KONTEN HALAMAN UTAMA - HEADER
# =====================================================================
st.markdown("<h1 class='main-title'>🏥 SISTEM PREDIKSI RISIKO DIABETES</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='sub-title'>Implementasi Klasifikasi Menggunakan Algoritma Naive Bayes</h4>", unsafe_allow_html=True)

# Notifikasi status model (asli atau simulasi)
if is_simulation:
    st.warning("⚠️ **Mode Demo Aktif:** File `model_diabetes.pkl` belum terdeteksi di folder proyek Anda. Sistem berjalan dalam mode simulasi cerdas untuk demonstrasi antarmuka. Silakan letakkan file hasil download dari Google Colab Anda di folder yang sama untuk beralih ke model asli.")
else:
    st.success("🎯 **Model Terhubung:** Otak kecerdasan buatan `model_diabetes.pkl` berhasil dimuat dengan sempurna!")

# Memisahkan halaman utama menjadi 2 Tab agar interaktif dan rapi
tab_form, tab_info = st.tabs(["📝 Formulir Analisis Medis", "📘 Detail Atribut & Evaluasi"])

# ---------------------------------------------------------------------
# TAB 1: FORMULIR INPUT MEDIS
# ---------------------------------------------------------------------
with tab_form:
    st.markdown("""
    Silakan masukkan parameter klinis pasien di bawah ini berdasarkan hasil laboratorium resmi.
    Sistem akan menganalisis probabilitas risiko menggunakan algoritma **Naive Bayes**.
    """)
    
    # Menggunakan Grid 2 Kolom untuk input data agar rapi di desktop & mobile
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Kehamilan (Pregnancies)", 
            min_value=0, max_value=20, value=1, step=1,
            help="Berapa kali pasien wanita telah hamil. Isi 0 jika pasien pria."
        )
        
        glucose = st.number_input(
            "Kadar Glukosa Plasma (Glucose) - mg/dL", 
            min_value=0.0, max_value=300.0, value=117.0, step=1.0,
            help="Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral."
        )
        
        blood_pressure = st.number_input(
            "Tekanan Darah Diastolik (Blood Pressure) - mmHg", 
            min_value=0.0, max_value=200.0, value=72.0, step=1.0,
            help="Tekanan darah saat jantung dalam kondisi relaksasi."
        )
        
        skin_thickness = st.number_input(
            "Ketebalan Lipatan Kulit (Skin Thickness) - mm", 
            min_value=0.0, max_value=100.0, value=29.0, step=1.0,
            help="Ketebalan lipatan kulit pada bagian otot trisep pasien."
        )

    with col2:
        insulin = st.number_input(
            "Kadar Insulin Serum (Insulin) - mu U/ml", 
            min_value=0.0, max_value=1000.0, value=125.0, step=1.0,
            help="Kadar hormon insulin puasa 2 jam dalam serum darah."
        )
        
        bmi = st.number_input(
            "Indeks Massa Tubuh (BMI) - kg/m²", 
            min_value=0.0, max_value=100.0, value=32.3, step=0.1,
            help="Rasio berat badan relatif terhadap tinggi badan: Berat badan (kg) / Tinggi badan (m)²."
        )
        
        pedigree = st.number_input(
            "Fungsi Silsilah Diabetes (Diabetes Pedigree Function)", 
            min_value=0.000, max_value=3.000, value=0.471, step=0.001, format="%.3f",
            help="Skor riwayat diabetes keluarga. Makin tinggi skor, makin besar faktor risiko keturunan."
        )
        
        age = st.number_input(
            "Usia (Age) - Tahun", 
            min_value=1, max_value=120, value=33, step=1,
            help="Usia kronologis pasien pada saat pemeriksaan medis dilakukan."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Eksekusi
    proses_analisis = st.button("🔍 PROSES EVALUASI MEDIS")
    
    if proses_analisis:
        st.write("---")
        st.markdown("### 📊 Hasil Klasifikasi Risiko")
        
        col_hasil_kiri, col_hasil_kanan = st.columns([2, 1])
        
        # Skenario prediksi menggunakan model asli atau simulasi matematis
        if not is_simulation:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            probabilitas = model.predict_proba(input_data)[0]
            persen_diabetes = probabilitas[1] * 100
            persen_sehat = probabilitas[0] * 100
        else:
            # Algoritma simulasi interaktif berdasarkan bobot medis jika file pkl belum ada
            skor_risiko = 0
            if glucose > 140: skor_risiko += 40
            elif glucose > 100: skor_risiko += 20
            
            if bmi > 30: skor_risiko += 25
            elif bmi > 25: skor_risiko += 15
            
            if age > 40: skor_risiko += 15
            if pedigree > 0.5: skor_risiko += 15
            if pregnancies > 3: skor_risiko += 5
            
            persen_diabetes = min(max(skor_risiko, 10), 95)
            persen_sehat = 100 - persen_diabetes

        # =====================================================================
        # LOGIKA PENENTUAN 3 KEMUNGKINAN (HIJAU, KUNING, MERAH)
        # =====================================================================
        with col_hasil_kiri:
            # 1. KEMUNGKINAN RISIKO TINGGI (KOTAK MERAH)
            if persen_diabetes > 65:
                st.markdown(f"""
                <div style="background-color: #7f1d1d; border-left: 5px solid #dc2626; padding: 15px; border-radius: 8px; color: #fef2f2; margin-bottom: 15px;">
                    <h4>⚠️ RISIKO TINGGI DIABETES ({persen_diabetes:.2f}%)</h4>
                    <strong>Sistem mendeteksi kecenderungan kuat pada pola data medis diabetes.</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                * **Rekomendasi:** Segera jadwalkan konsultasi dengan Dokter Spesialis Penyakit Dalam dan lakukan uji lab HbA1c. Pusatkan perhatian pada pembatasan ketat gula murni.
                """)
                
            # 2. KEMUNGKINAN RISIKO SEDANG / WASPADA (KOTAK KUNING)
            elif 40 <= persen_diabetes <= 65:
                st.markdown(f"""
                <div style="background-color: #7c2d12; border-left: 5px solid #ea580c; padding: 15px; border-radius: 8px; color: #fff7ed; margin-bottom: 15px;">
                    <h4>🟡 RISIKO SEDANG / WASPADA ({persen_diabetes:.2f}%)</h4>
                    <strong>Kondisi berada di zona abu-abu (Prediabetes). Beberapa indikator tubuh mulai tidak seimbang.</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                * **Rekomendasi:** Kondisi ini sangat bisa dibalikkan menjadi normal! Mulailah menurunkan berat badan (karena faktor BMI Anda yang tinggi), kurangi porsi karbohidrat olahan, dan tingkatkan aktivitas fisik harian.
                """)
                
            # 3. KEMUNGKINAN RISIKO AMAN (KOTAK HIJAU)
            else:
                st.markdown(f"""
                <div style="background-color: #064e3b; border-left: 5px solid #059669; padding: 15px; border-radius: 8px; color: #ecfdf5; margin-bottom: 15px;">
                    <h4>✅ RISIKO RENDAH / AMAN ({persen_sehat:.2f}%)</h4>
                    <strong>Indikator klinis Anda secara umum berada dalam batas normal.</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                * **Rekomendasi:** Pertahankan pola hidup sehat dan aktif Anda saat ini. Lakukan kontrol gula darah berkala minimal setahun sekali.
                """)
                
        with col_hasil_kanan:
            st.markdown("**Detail Proporsi Keputusan:**")
            st.metric(label="Rendah / Aman", value=f"{persen_sehat:.1f}%")
            st.metric(label="Tinggi (Diabetes)", value=f"{persen_diabetes:.1f}%")

# ---------------------------------------------------------------------
# TAB 2: GLOSARIUM FITUR & EVALUASI KINERJA MODEL
# ---------------------------------------------------------------------
with tab_info:
    st.markdown("### 📋 Glosarium Fitur Dataset")
    st.write("Model ini menggunakan 8 indikator medis klinis yang dilengkapi batas angka analisis minimal dan maksimal untuk penentuan risiko:")
    
    # Penambahan kolom batas angka Aman, Sedang, dan Tinggi secara mendalam sesuai instruksi
    kamus_fitur = {
        "Nama Fitur": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "Jenis Pengukuran": [
            "Numerik (Kali)", "mg/dL (Glukosa Plasma)", "mmHg (Tekanan Diastolik)", 
            "mm (Lipatan Kulit)", "mu U/ml (Kadar Insulin)", "kg/m² (Berat/Tinggi)", 
            "Skor Riwayat Keluarga", "Tahun (Usia Pasien)"
        ],
        "Penjelasan Medis": [
            "Frekuensi kehamilan yang dialami pasien wanita sepanjang masa suburnya.",
            "Kadar konsentrasi gula dalam plasma darah 2 jam setelah tes beban glukosa oral.",
            "Tekanan darah diastolik (tekanan pada pembuluh darah saat jantung rileks).",
            "Ketebalan lipatan kulit pada bagian trisep lengan atas untuk estimasi lemak tubuh.",
            "Kadar hormon insulin dalam darah setelah pemberian asupan glukosa.",
            "Indeks Massa Tubuh pasien (Berat badan dalam kg dibagi tinggi dalam meter kuadrat).",
            "Skor kalkulasi silsilah keluarga yang memetakan faktor risiko diabetes secara genetik.",
            "Usia kronologis pasien pada saat pemeriksaan medis dilakukan."
        ],
        "Aman (🟢)": [
            "0 - 2 kali",
            "< 100 mg/dL",
            "60 - 80 mmHg",
            "< 20 mm",
            "16 - 100 mu U/ml",
            "18.5 - 24.9 kg/m²",
            "< 0.300",
            "21 - 30 Tahun"
        ],
        "Sedang / Waspada (🟡)": [
            "3 kali",
            "100 - 139 mg/dL",
            "81 - 89 mmHg",
            "20 - 29 mm",
            "101 - 159 mu U/ml",
            "25.0 - 29.9 kg/m²",
            "0.300 - 0.499",
            "31 - 39 Tahun"
        ],
        "Tinggi / Bahaya (🔴)": [
            "> 3 kali",
            "≥ 140 mg/dL",
            "≥ 90 mmHg",
            "≥ 30 mm",
            "≥ 160 mu U/ml (Atau mendekati 0)",
            "≥ 30.0 kg/m²",
            "≥ 0.500",
            "≥ 40 Tahun"
        ]
    }
    st.table(pd.DataFrame(kamus_fitur))
    
    st.write("---")
    st.markdown("### 📈 Matriks Evaluasi Model Naive Bayes")
    
    col_acc, col_prec, col_rec, col_f1 = st.columns(4)
    with col_acc:
        st.metric(label="Akurasi Pengujian", value="75.32%")
    with col_prec:
        st.metric(label="Presisi (Precision)", value="64.91%")
    with col_rec:
        st.metric(label="Sensitivitas (Recall)", value="67.27%")
    with col_f1:
        st.metric(label="F1-Score", value="66.07%")
        
    st.markdown("""
    **Penjelasan Distribusi Kebenaran Prediksi (*Confusion Matrix*):**
    * **79 Pasien (True Negative):** Pasien sehat yang berhasil ditebak Sehat dengan akurat oleh sistem.
    * **37 Pasien (True Positive):** Pasien penderita diabetes yang berhasil terdeteksi dengan benar oleh sistem.
    * **18 Pasien (False Negative):** Pasien diabetes yang terlewat atau terdiagnosis aman (salah tebak).
    * **20 Pasien (False Positive):** Pasien sehat yang keliru dicurigai mengalami diabetes oleh sistem.
    """)

# =====================================================================
# FOOTER DISKLAIMER MEDIS (PENTING UNTUK STANDAR ETIKA PENELITIAN)
# =====================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.warning("""
🚨 **Disklaimer Akademis:** Sistem ini dikembangkan sebagai prototipe riset teknologi informasi berbasis penambangan data (*data mining*). Hasil analisis dari aplikasi ini **tidak boleh dijadikan acuan diagnosis medis mandiri klinis yang bersifat mutlak**. Diagnosis medis yang sah dan profesional hanya boleh dikeluarkan oleh dokter atau tenaga kesehatan berwenang melalui pemeriksaan medis langsung.
""")    