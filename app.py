import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ======================================================================================
# KONFIGURASI HALAMAN & JUDUL
# ======================================================================================
st.set_page_config(
    page_title="Prediksi Kepribadian Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# FUNGSI-FUNGSI UTAMA
# ======================================================================================

@st.cache_data
def load_data():
    """Fungsi untuk memuat dan membersihkan data dari file CSV."""
    df = pd.read_csv('personality_dataset.csv')
    df_processed = df.copy()

    # Menangani nilai yang hilang (NaN)
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
            else:
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)

    # Menggunakan LabelEncoder untuk mengubah fitur kategorikal menjadi angka
    le = LabelEncoder()
    for col in ['Stage_fear', 'Drained_after_socializing', 'Personality']:
        df_processed[col] = le.fit_transform(df_processed[col])

    return df, df_processed

@st.cache_resource
def train_models(data):
    """Fungsi untuk melatih model dan mengembalikan hasil evaluasi."""
    X = data.drop('Personality', axis=1)
    y = data['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, target_names=['Introvert', 'Extrovert'])
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # --- Naive Bayes ---
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb, target_names=['Introvert', 'Extrovert'])
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    return knn, nb, accuracy_knn, report_knn, cm_knn, accuracy_nb, report_nb, cm_nb

# --- FITUR BARU: Fungsi untuk menghitung profil rata-rata ---
@st.cache_data
def get_average_profiles(_df_processed):
    """Menghitung profil rata-rata untuk Introvert dan Extrovert."""
    introvert_profile = _df_processed[_df_processed['Personality'] == 0].mean()
    extrovert_profile = _df_processed[_df_processed['Personality'] == 1].mean()
    return introvert_profile.drop('Personality'), extrovert_profile.drop('Personality')

# ======================================================================================
# MEMUAT DATA, MELATIH MODEL, DAN MENGHITUNG PROFIL
# ======================================================================================
df_raw, df_processed = load_data()
model_knn, model_nb, acc_knn, report_knn, cm_knn, acc_nb, report_nb, cm_nb = train_models(df_processed)
introvert_avg, extrovert_avg = get_average_profiles(df_processed)

# ======================================================================================
# SIDEBAR / NAVIGASI
# ======================================================================================
st.sidebar.title("Navigasi 🧭")
page = st.sidebar.radio("Pilih Halaman:", ["Analisis Data (EDA)", "Hasil Pelatihan Model", "Lakukan Prediksi"])

# ======================================================================================
# HALAMAN 1: Exploratory Data Analysis (EDA)
# ======================================================================================
if page == "Analisis Data (EDA)":
    st.title("📊 Analisis Data Eksplorasi (EDA)")
    st.markdown("Halaman ini menampilkan analisis awal dari dataset kepribadian. Anda dapat melihat data mentah, statistik deskriptif, dan berbagai visualisasi untuk memahami karakteristik data.")
    st.divider()

    st.subheader("Tabel Dataset")
    st.dataframe(df_raw)

    st.subheader("Statistik Deskriptif")
    st.write(df_raw.describe())
    
    st.divider()

    st.subheader("Visualisasi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribusi Kepribadian")
        fig_pie = px.pie(df_raw, names='Personality', title='Persentase Introvert vs. Extrovert', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("#### Distribusi Rasa Takut Panggung")
        fig_bar = px.bar(df_raw['Stage_fear'].value_counts(), x=df_raw['Stage_fear'].value_counts().index, y=df_raw['Stage_fear'].value_counts().values, title='Jumlah Responden Berdasarkan Rasa Takut Panggung', labels={'x':'Rasa Takut Panggung', 'y':'Jumlah Orang'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Hubungan Antara Lingkaran Pertemanan dan Kehadiran di Acara Sosial")
    fig_scatter = px.scatter(df_raw, x='Friends_circle_size', y='Social_event_attendance', color='Personality', title='Ukuran Lingkaran Teman vs. Kehadiran di Acara Sosial', labels={'Friends_circle_size': 'Ukuran Lingkaran Teman', 'Social_event_attendance': 'Kehadiran di Acara Sosial'}, hover_data=['Time_spent_Alone'])
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- FITUR BARU: Peta Korelasi ---
    st.divider()
    st.subheader("Peta Korelasi Antar Fitur")
    st.markdown("Heatmap ini menunjukkan bagaimana setiap fitur numerik berhubungan satu sama lain. Angka mendekati 1 (biru tua) atau -1 (merah tua) menunjukkan korelasi yang kuat.")
    corr = df_processed.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# ======================================================================================
# HALAMAN 2: HASIL PELATIHAN MODEL
# ======================================================================================
elif page == "Hasil Pelatihan Model":
    st.title("🤖 Hasil Pelatihan Model Machine Learning")
    st.markdown("Di halaman ini, kita melihat performa dari dua model yang telah dilatih: **K-Nearest Neighbors (KNN)** dan **Gaussian Naive Bayes**.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.metric(label="Akurasi Model", value=f"{acc_knn:.2%}")
        st.text("Laporan Klasifikasi:")
        st.code(report_knn)

        # --- FITUR BARU: Confusion Matrix ---
        st.text("Confusion Matrix:")
        fig_cm_knn, ax_cm_knn = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Introvert', 'Extrovert'], yticklabels=['Introvert', 'Extrovert'], ax=ax_cm_knn)
        ax_cm_knn.set_xlabel('Prediksi')
        ax_cm_knn.set_ylabel('Aktual')
        st.pyplot(fig_cm_knn)

    with col2:
        st.subheader("Gaussian Naive Bayes")
        st.metric(label="Akurasi Model", value=f"{acc_nb:.2%}")
        st.text("Laporan Klasifikasi:")
        st.code(report_nb)

        # --- FITUR BARU: Confusion Matrix ---
        st.text("Confusion Matrix:")
        fig_cm_nb, ax_cm_nb = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', xticklabels=['Introvert', 'Extrovert'], yticklabels=['Introvert', 'Extrovert'], ax=ax_cm_nb)
        ax_cm_nb.set_xlabel('Prediksi')
        ax_cm_nb.set_ylabel('Aktual')
        st.pyplot(fig_cm_nb)

# ======================================================================================
# HALAMAN 3: LAKUKAN PREDIKSI
# ======================================================================================
elif page == "Lakukan Prediksi":
    st.title("🔮 Formulir Prediksi Kepribadian")
    st.markdown("Isi formulir di bawah ini dengan kebiasaan Anda, dan model akan mencoba memprediksi apakah Anda cenderung **Introvert** atau **Extrovert**.")
    st.divider()

    with st.form("prediction_form"):
        st.subheader("Isi Data Anda:")
        col1, col2 = st.columns(2)
        with col1:
            time_spent_alone = st.slider("Waktu Dihabiskan Sendiri (Jam per hari)", 0, 11, 5)
            stage_fear = st.radio("Apakah Anda memiliki demam panggung?", ("Ya", "Tidak"))
            social_event_attendance = st.slider("Frekuensi menghadiri acara sosial (0-10)", 0, 10, 5)
            going_outside = st.slider("Frekuensi keluar rumah (kali per minggu)", 0, 7, 3)
        with col2:
            drained_after_socializing = st.radio("Apakah Anda merasa lelah setelah bersosialisasi?", ("Ya", "Tidak"))
            friends_circle_size = st.slider("Jumlah teman dekat (0-15)", 0, 15, 5)
            post_frequency = st.slider("Frekuensi posting di media sosial (0-10)", 0, 10, 4)
        
        submit_button = st.form_submit_button(label="✨ Lakukan Prediksi!")

    if submit_button:
        stage_fear_num = 1 if stage_fear == "Ya" else 0
        drained_after_socializing_num = 1 if drained_after_socializing == "Ya" else 0
        input_data = pd.DataFrame([[time_spent_alone, stage_fear_num, social_event_attendance, going_outside, drained_after_socializing_num, friends_circle_size, post_frequency]], columns=df_processed.drop('Personality', axis=1).columns)

        # --- FITUR BARU: Prediksi dengan Probabilitas ---
        prediction_knn = model_knn.predict(input_data)
        proba_knn = model_knn.predict_proba(input_data)
        confidence_knn = proba_knn[0][prediction_knn[0]]
        result_knn = "Extrovert" if prediction_knn[0] == 1 else "Introvert"

        prediction_nb = model_nb.predict(input_data)
        proba_nb = model_nb.predict_proba(input_data)
        confidence_nb = proba_nb[0][prediction_nb[0]]
        result_nb = "Extrovert" if prediction_nb[0] == 1 else "Introvert"

        st.divider()
        st.subheader("🎉 Hasil Prediksi Anda:")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info(f"**Prediksi Model KNN:** Anda cenderung seorang **{result_knn}** (kepercayaan: {confidence_knn:.0%})")
        with col_res2:
            st.success(f"**Prediksi Model Naive Bayes:** Anda cenderung seorang **{result_nb}** (kepercayaan: {confidence_nb:.0%})")

        # --- FITUR BARU: Radar Chart ---
        with st.expander("Lihat Perbandingan Profil Anda (Radar Chart)"):
            feature_names = list(extrovert_avg.index)
            max_values = df_processed.drop('Personality', axis=1).max()
            user_normalized = input_data.iloc[0] / max_values
            introvert_normalized = introvert_avg / max_values
            extrovert_normalized = extrovert_avg / max_values

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=user_normalized, theta=feature_names, fill='toself', name='Profil Anda'))
            fig_radar.add_trace(go.Scatterpolar(r=introvert_normalized, theta=feature_names, fill='toself', name='Rata-rata Introvert', opacity=0.7))
            fig_radar.add_trace(go.Scatterpolar(r=extrovert_normalized, theta=feature_names, fill='toself', name='Rata-rata Extrovert', opacity=0.7))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Perbandingan Profil Anda dengan Profil Rata-Rata")
            st.plotly_chart(fig_radar, use_container_width=True)
