import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from src.data_loader import load_raw_data, merge_datasets
from src.feature_engineering import process_features
from src.model_trainer import train_turnover_model
from src.predictor import predict_personnel_risk, get_actionable_recommendations

st.set_page_config(page_title="Analitik Risiko Resign Fieldforce", layout="wide", page_icon="📊")

st.title("📊 Analitik Risiko Resign Fieldforce")
st.markdown("Prediksi probabilitas seorang karyawan untuk keluar (*resign/turnover*) berdasarkan beban kerja harian, kepatuhan kunjungan, waktu kerja efektif, dan jam lembur.")

# --- SIDEBAR ---
st.sidebar.header("📁 Unggah Data & Filter")
st.sidebar.markdown("Silakan unggah file Excel Anda, atau gunakan data sampel bawaan jika ingin menguji aplikasi.")

# Tombol untuk memuat data sampel
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

if st.sidebar.button("⚙️ Gunakan Data Sampel"):
    st.session_state.use_sample = True

absen_file = st.sidebar.file_uploader("Unggah Data Absen.xlsx", type=['xlsx'])
visit_file = st.sidebar.file_uploader("Unggah Data Visit.xlsx", type=['xlsx'])

import os
# Jika tombol ditekan, override uploader dengan file lokal
if st.session_state.use_sample:
    if os.path.exists('Data Absen.xlsx') and os.path.exists('Data Visit.xlsx'):
        absen_file = 'Data Absen.xlsx'
        visit_file = 'Data Visit.xlsx'
        st.sidebar.success("✅ Data sampel aktif!")
    else:
        st.sidebar.error("File sampel tidak ditemukan di dalam direktori.")

if not absen_file or not visit_file:
    st.info("👈 Harap unggah file `Data Absen.xlsx` dan `Data Visit.xlsx` di sidebar, atau klik tombol **'⚙️ Gunakan Data Sampel'** untuk melanjutkan.")
    st.stop()

# Load Data
with st.spinner("Memuat dan memproses data..."):
    df_absen, df_visit = load_raw_data(absen_file, visit_file)
    if df_absen.empty or df_visit.empty:
        st.error("Gagal memuat data. Pastikan file yang diunggah benar.")
        st.stop()

    df_merged = merge_datasets(df_absen, df_visit)
    df_features = process_features(df_merged)

    # Train Model
    model_data = train_turnover_model(df_features)

# Filters
st.sidebar.subheader("Filter")
if 'Group Personil' in df_features.columns:
    group_filter = st.sidebar.selectbox("Filter berdasarkan Group Personil", ["Semua"] + list(df_features['Group Personil'].dropna().unique()))
    if group_filter != "Semua":
        df_display = df_features[df_features['Group Personil'] == group_filter]
    else:
        df_display = df_features
else:
    df_display = df_features

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Operasional", "🔮 Evaluator Risiko Individu", "🧠 Performa Model ML", "ℹ️ Tentang Aplikasi"])

# TAB 1: Operational Overview
with tab1:
    st.subheader("Ringkasan Operasional & KPI Fieldforce")
    
    col1, col2, col3, col4 = st.columns(4)
    total_personnel = len(df_display['Personil Code'].unique())
    total_visits = df_display['Visited'].sum() if 'Visited' in df_display else 0
    avg_work_dur = (df_display['Total Jam Absen'].mean() / 60) if 'Total Jam Absen' in df_display else 0
    avg_overtime = df_display['Overtime_Hours'].mean() if 'Overtime_Hours' in df_display else 0
    
    col1.metric("Total Personel", f"{total_personnel}")
    col2.metric("Total Kunjungan Selesai", f"{total_visits:,.0f}")
    col3.metric("Rata-rata Durasi Kerja (Jam)", f"{avg_work_dur:.1f}j")
    col4.metric("Rata-rata Lembur (Jam)", f"{avg_overtime:.2f}j")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Rasio Kerja Efektif vs Lembur**")
        fig1 = px.scatter(
            df_display, x='Effective_Work_Ratio', y='Overtime_Hours', 
            color='Churn_Target', hover_data=['Personil Name'],
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.markdown("**Distribusi Kepatuhan Kunjungan (Route Compliance)**")
        fig2 = px.histogram(df_display, x='Route_Compliance', nbins=20, color_discrete_sequence=['#F05A28'])
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: Individual Risk Evaluator
with tab2:
    st.subheader("Evaluator Risiko Resign Personel Individu")
    
    personil_list = df_display['Personil Name'].dropna().unique().tolist()
    selected_personil = st.selectbox("Pilih Personel", personil_list)
    
    if selected_personil:
        personil_data = df_display[df_display['Personil Name'] == selected_personil].iloc[0]
        
        # Prepare input for prediction
        input_features = {
            'Overtime_Hours': personil_data['Overtime_Hours'],
            'Effective_Work_Ratio': personil_data['Effective_Work_Ratio'],
            'Route_Compliance': personil_data['Route_Compliance'],
            'Workload_Volume': personil_data['Workload_Volume'],
            'Avg_Visit_Duration': personil_data['Avg_Visit_Duration']
        }
        
        risk_res = predict_personnel_risk(model_data, input_features)
        prob = risk_res['probability']
        risk_level = risk_res['risk_level']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Risiko Resign: {risk_level}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_level == 'Tinggi' else "orange" if risk_level == 'Sedang' else "green"},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.markdown("**Rincian Metrik (vs Rata-rata Tim)**")
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Lembur (Jam)", f"{input_features['Overtime_Hours']:.2f}", f"{(input_features['Overtime_Hours'] - df_display['Overtime_Hours'].mean()):.2f}", delta_color="inverse")
            m_col2.metric("Kerja Efektif (%)", f"{input_features['Effective_Work_Ratio']:.1f}%", f"{(input_features['Effective_Work_Ratio'] - df_display['Effective_Work_Ratio'].mean()):.1f}%")
            
            m_col3, m_col4 = st.columns(2)
            m_col3.metric("Kepatuhan Rute (%)", f"{input_features['Route_Compliance']:.1f}%", f"{(input_features['Route_Compliance'] - df_display['Route_Compliance'].mean()):.1f}%")
            m_col4.metric("Beban Kerja (Item)", f"{input_features['Workload_Volume']:.0f}", f"{(input_features['Workload_Volume'] - df_display['Workload_Volume'].mean()):.0f}", delta_color="inverse")
            
        st.markdown("### Rekomendasi Tindakan HRD")
        recs = get_actionable_recommendations(risk_level, input_features)
        for rec in recs:
            if risk_level == 'Tinggi':
                st.error(rec)
            elif risk_level == 'Sedang':
                st.warning(rec)
            else:
                st.success(rec)

# TAB 3: ML Model Performance
with tab3:
    st.subheader("Performa Model ML & Penggerak Fitur")
    
    if model_data:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Akurasi", f"{model_data['metrics']['Accuracy']:.2%}")
        m_col2.metric("Skor-F1", f"{model_data['metrics']['F1-Score']:.2f}")
        m_col3.metric("ROC-AUC", f"{model_data['metrics']['ROC-AUC']:.2f}")
        
        st.markdown("---")
        
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            st.markdown("**Tingkat Kepentingan Fitur (Feature Importance)**")
            fig_imp = px.bar(
                model_data['feature_importance'], 
                x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues'
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_f2:
            st.markdown("**Ekspor Data**")
            st.write("Unduh daftar seluruh personel beserta fitur dan target prediksi risiko resign mereka.")
            
            # Export to Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_display.to_excel(writer, index=False, sheet_name='Data_Resign')
            processed_data = output.getvalue()
            
            st.download_button(
                label="📥 Unduh Excel",
                data=processed_data,
                file_name="Analitik_Resign_HRD.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Model tidak dilatih karena data tidak ditemukan.")

# TAB 4: Tentang Aplikasi
with tab4:
    st.subheader("ℹ️ Tentang Aplikasi Analitik Risiko Resign")
    
    st.markdown("""
    ### 📌 Judul Ide Solusi
    **"FMCG Fieldforce Churn Analytics: Sistem Peringatan Dini Berbasis Machine Learning untuk Optimalisasi Kinerja dan Retensi Karyawan Lapangan"**
    
    ---

    Aplikasi **Analitik Risiko Resign (Turnover) Fieldforce** ini dirancang secara khusus untuk memodernisasi cara kerja tim HRD dalam mengelola karyawan lapangan.

    ### 📖 Latar Belakang Masalah
    Dalam industri FMCG (Fast-Moving Consumer Goods), divisi lapangan seperti *Merchandiser*, *Sales Canvasser*, dan *Field Force* merupakan ujung tombak perusahaan. Namun, posisi ini seringkali memiliki **tingkat perputaran karyawan (turnover/resign) yang sangat tinggi**. 
    
    Permasalahan utama yang sering terjadi di lapangan meliputi:
    1. **Beban Kerja yang Tidak Terlihat (Blind Spot):** Manajemen seringkali kesulitan mengukur seberapa berat beban kerja harian di lapangan. Seorang karyawan mungkin memiliki target kunjungan toko yang terlihat wajar di atas kertas, namun kondisi riil di lapangan (kemacetan, jarak antar toko, waktu tunggu) membuat mereka harus melakukan lembur yang tidak dibayar secara berlebihan (*unpaid overtime*), yang berujung pada kelelahan ekstrem (*burnout*).
    2. **Kurangnya Pengambilan Keputusan Berbasis Data:** Keputusan HRD terkait retensi karyawan seringkali bersifat reaktif—HRD baru bertindak *setelah* menerima surat pengunduran diri dari karyawan potensial. Tidak ada sistem yang mengolah data operasional harian (seperti durasi kunjungan dan efektivitas kerja) menjadi alarm peringatan dini.
    3. **Distribusi Area yang Tidak Seimbang:** Karyawan seringkali kehilangan motivasi (*demotivated*) ketika rasio jam kerja efektif mereka rendah akibat rute yang tidak dioptimalkan, membuat mereka merasa membuang banyak waktu di jalan tanpa mencapai target.

    Oleh karena itu, diperlukan sebuah sistem cerdas yang mampu menganalisis pola operasional harian ini secara prediktif untuk menyelamatkan aset SDM terbaik perusahaan sebelum mereka memutuskan untuk keluar.

    ### 🔍 Analisa Masalah (Metode: 5 Whys)
    Untuk menggali akar permasalahan (Root Cause) dari tingginya angka *turnover* ini, kita menggunakan metode **5 Whys**:
    - **Problem Utama:** Tingginya tingkat pengunduran diri (*resign*) pada karyawan lapangan.
    - **Why 1?** Mengapa banyak karyawan lapangan yang *resign*?
      *Jawaban:* Karena mereka mengalami stres, kelelahan fisik yang ekstrem (*burnout*), dan demotivasi.
    - **Why 2?** Mengapa mereka mengalami *burnout* dan demotivasi?
      *Jawaban:* Karena mereka terpaksa melakukan lembur (*overtime*) hampir setiap hari, namun target kunjungan (*Route Compliance*) seringkali tetap gagal tercapai.
    - **Why 3?** Mengapa mereka harus lembur dan target tetap gagal, padahal beban kerja sudah dihitung oleh manajemen?
      *Jawaban:* Karena "Waktu Kerja Efektif" mereka sangat rendah. Banyak waktu produktif yang terbuang di jalan akibat rute yang tidak efisien, jarak antar toko yang terlalu jauh, dan kemacetan.
    - **Why 4?** Mengapa rute kunjungan dan beban kerja karyawan tidak efisien?
      *Jawaban:* Karena pembagian teritori kerja (*Workload Volume*) hanya didasarkan pada asumsi statis di atas kertas, tanpa mempertimbangkan kondisi riil dan kapasitas harian di lapangan.
    - **Why 5?** Mengapa manajemen/HRD mengambil keputusan hanya berdasarkan asumsi, bukan data riil?
      *Jawaban:* (Akar Masalah) **Karena tidak adanya alat atau sistem analitik prediktif** yang mampu secara otomatis mengolah data mentah absensi dan pelacakan GPS (*Data Visit*) menjadi peringatan dini (*Early Warning System*) bagi HRD.

    ### 📊 Data Pendukung
    Untuk membuktikan dan menyelesaikan permasalahan di atas, aplikasi ini menggunakan sampel dokumentasi data operasional lapangan riil (dapat diunggah melalui *sidebar*):
    - **`Data Absen.xlsx`**: Berisi catatan otentik terkait *Total Jam Absen* dan *Total Jam Efektif (JEM)*. Dari data ini, kita menemukan fakta bahwa banyak karyawan menghabiskan waktu lebih dari standar 8 jam kerja, namun rasio efektivitas kerjanya rendah akibat waktu tunggu atau perjalanan.
    - **`Data Visit.xlsx`**: Berisi catatan geolokasi (*Latitude/Longitude*), durasi kunjungan per titik (*Interval*), dan volume beban kerja harian (*Jumlah Activity Items*). Data ini menjadi bukti empiris adanya area atau rute tertentu yang memiliki *workload* ekstrem.

    > **🌐 Live Deployment URL:** [Akan dicantumkan di sini setelah aplikasi di-deploy / di-hosting]

    ### 🎯 Tujuan Utama
    Menciptakan sebuah **Sistem Peringatan Dini (Early Warning System)** dengan mengubah data absensi harian dan data aktivitas kunjungan lapangan menjadi metrik terukur. Dengan ini, HRD dapat memprediksi secara akurat karyawan mana yang berpotensi kelelahan (*burnout*) atau berniat *resign* karena beban operasional.


    ### 🌟 Manfaat untuk Perusahaan & HRD
    - **Proaktif, Bukan Reaktif:** Melakukan intervensi pencegahan (diskusi 1-on-1 / *coaching*) sebelum karyawan terbaik benar-benar keluar.
    - **Distribusi Beban Kerja Adil:** Melihat secara transparan jika ada tim yang mengalami *overwork* (lembur berlebih / rute terlalu padat) untuk disesuaikan.
    - **Efisiensi Biaya:** Menurunkan angka *turnover* untuk menghemat biaya operasional perusahaan (biaya rekrutmen, waktu *training*, hilangnya potensi *sales*).
    - **Rekomendasi Otomatis:** Sistem secara pintar mencetak solusi konkret berdasarkan pola masalah karyawan terkait.

    ### ⚙️ Arsitektur Teknis
    Aplikasi ini dibangun menggunakan arsitektur *Machine Learning* modern:
    
    1. **Tampilan Antarmuka (UI/Dashboard):** Dibangun menggunakan **Streamlit** (Python) untuk interaktivitas data yang cepat dan elegan.
    2. **Pemrosesan Data (Pandas):** 
       - Otomatis membaca data mentah `Data Absen.xlsx` & `Data Visit.xlsx`.
       - Melakukan *Data Merging* (Left Join) berdasarkan `Personil Code`.
    3. **Feature Engineering (Penciptaan Metrik):**
       - **Lembur (Overtime):** Jam kerja total dikurangi standar 8 jam.
       - **Kepatuhan Rute (Route Compliance):** Rasio toko yang berhasil dikunjungi vs target.
       - **Rasio Kerja Efektif:** Persentase jam mereka benar-benar bekerja (*Effective Call*) dibanding sekadar berada di lapangan.
    4. **Algoritma Machine Learning (AI Model):**
       - Menggunakan **Random Forest Classifier** (*Scikit-Learn*). Sangat ideal karena memiliki fitur *Feature Importance* untuk menjelaskan ke HRD faktor apa yang paling besar pengaruhnya terhadap risiko resign.
    5. **Visualisasi Data:** Menggunakan **Plotly** untuk mencetak grafik interaktif (*Scatter plot*, *Histogram*, *Gauge Indicator*).

    ### 💡 Alternatif Solusi (Solusi yang Diusulkan)
    Berdasarkan akar permasalahan (*Root Cause*) yang telah dianalisis, solusi utama yang kami angkat adalah membangun **Dashboard Analitik HR Prediktif Berbasis Machine Learning**. Solusi ini dirancang dengan detail sebagai berikut:
    
    1. **Otomatisasi Peringatan Dini (AI-Driven Early Warning):**
       Mengintegrasikan model kecerdasan buatan (*Random Forest Classifier*) yang mampu memproses data mentah harian karyawan. Model ini akan secara otomatis memberikan "Skor Risiko Resign" (Rendah, Sedang, Tinggi) untuk setiap karyawan tanpa perlu dianalisis secara manual satu per satu.
    2. **Pemetaan Matriks Kinerja vs Kelelahan (Burnout Matrix):**
       Solusi ini tidak hanya melihat "siapa yang bekerja paling keras", tetapi memvisualisasikan korelasi antara **Jam Lembur** vs **Kerja Efektif**. Hal ini memberikan visibilitas penuh kepada manajemen untuk melihat secara adil area mana yang membutuhkan tambahan tenaga kerja, dan area mana yang rutenya perlu dirombak ulang.
    3. **Generator Rekomendasi Tindakan Cerdas (Actionable Insight Generator):**
       Solusi ini mengubah *dashboard* biasa menjadi asisten HRD cerdas. Jika seorang karyawan terdeteksi memiliki risiko *resign* tinggi akibat rute yang padat, sistem akan langsung merekomendasikan tindakan spesifik untuk Manajer (misal: *"Kepatuhan rute rendah, diskusikan kendala lalu lintas dan pertimbangkan restrukturisasi teritori"*), sehingga penanganan yang diberikan tepat sasaran sebelum karyawan tersebut kehabisan motivasi.

    ### 🚀 Rencana & Tahapan Implementasi
    Karena aplikasi MVP (*Minimum Viable Product*) ini sudah berhasil kami bangun, berikut adalah dokumentasi tahapan implementasi yang telah kami jalankan beserta rencana tahap akhirnya:

    **Tahap 1: Pengumpulan & Standarisasi Data (Selesai)**
    - *Aktivitas:* Mengambil sampel raw data operasional dari 2 sumber utama (`Data Absen.xlsx` dan `Data Visit.xlsx`).
    - *Hasil:* Script `data_loader.py` berhasil dibangun untuk menyatukan miliaran baris data kunjungan dan mencocokkannya (*Left Join*) dengan data absensi berdasarkan ID Karyawan (`Personil Code`).

    **Tahap 2: Rekayasa Fitur & Pelatihan Model AI (Selesai)**
    - *Aktivitas:* Mengubah data mentah menjadi metrik bisnis dan melatih algoritma Machine Learning.
    - *Hasil:* Script `feature_engineering.py` sukses merumuskan persentase *Route Compliance*, Rasio Efektivitas, dan Durasi Lembur. Algoritma **Random Forest** (di `model_trainer.py`) berhasil dilatih dan mampu mendeteksi korelasi fitur yang paling berpotensi menyebabkan karyawan *resign*.

    **Tahap 3: Pengembangan Dashboard Antarmuka (Selesai)**
    - *Aktivitas:* Membangun UI/UX menggunakan framework **Streamlit** agar mudah digunakan oleh staf HRD non-teknis.
    - *Hasil:* *Dashboard* interaktif (`app.py`) dengan 4 Tab utama berhasil beroperasi. Dilengkapi dengan filter Area/Grup, grafik matriks *Burnout*, *Speedometer* risiko individu, dan tombol *Download Report* Excel.

    **Tahap 4: Pengujian Lokal & Evaluasi (Selesai)**
    - *Aktivitas:* Simulasi pemrosesan data riil secara *offline/local*.
    - *Hasil:* Sistem telah berjalan stabil tanpa *error* di *localhost:8501*. Model AI terbukti mampu memberikan akurasi skor yang relevan dengan kondisi lapangan, dibuktikan dengan munculnya nama-nama kandidat *high-risk* yang masuk akal beserta rekomendasinya.

    **Tahap 5: Deployment Cloud & Sosialisasi (Rencana Selanjutnya)**
    - *Aktivitas:* Mengunggah sistem ini ke server publik (*Cloud Hosting* seperti AWS / Streamlit Community Cloud) agar dapat diakses dari mana saja via URL web. Dilanjutkan dengan sesi *training* penggunaan kepada tim HRD dan Manajer Area.
    - *Hasil yang Diharapkan:* Adopsi teknologi secara penuh oleh tim operasional untuk menekan angka *turnover* karyawan di kuartal berikutnya.

    ### 📅 Timeline Plan
    Berikut adalah estimasi timeline pengerjaan (*project roadmap*) yang kami lalui mulai dari fase inisiasi hingga tahap rilis:

    | Waktu | Fase / Aktivitas | Status | Keterangan |
    |---|---|---|---|
    | **Minggu 1** | Requirement Gathering & Data Extraction | ✅ Selesai | Diskusi dengan HRD untuk mendefinisikan *stress triggers* (Overtime, Workload, Efektivitas) dan ekstraksi raw data Excel. |
    | **Minggu 2** | Data Cleaning & Feature Engineering | ✅ Selesai | Membersihkan anomali data (misal: data kosong, tipe teks pada angka) dan meracik metrik matematis dari data harian. |
    | **Minggu 3** | AI Model Training | ✅ Selesai | Membangun dan melatih model *Random Forest Classifier*, serta melakukan *tuning* agar akurasi prediksi (*ROC-AUC*) optimal. |
    | **Minggu 4** | Dashboard Development (UI/UX) | ✅ Selesai | Mengintegrasikan model AI ke dalam *dashboard* Streamlit yang interaktif (Tab Operasional, Evaluator Individu, dan Report). |
    | **Minggu 5** | Cloud Deployment | ⏳ Berjalan | Menyiapkan *server hosting* agar aplikasi dapat diakses publik via tautan web. |
    | **Minggu 6** | Sosialisasi & Evaluasi | 🗓️ Terjadwal | *Training* ke pengguna akhir (Manajer Area / Tim Operasional) dan evaluasi dampaknya terhadap penurunan angka resign bulanan. |
    """)
