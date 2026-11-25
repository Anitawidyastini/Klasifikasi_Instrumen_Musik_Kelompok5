
import os
import json
import tempfile
import streamlit as st

from config import HASIL_DIR
from feature_extraction import ekstraksi_fitur_audio
from ann import forward_propagation


# ==========================================
# 1. KONFIGURASI PATH
# ==========================================

FOLDER_MODEL = os.path.join(HASIL_DIR, "model")
FOLDER_PREPROCESSING = os.path.join(HASIL_DIR, "preprocessing")

FILE_MODEL_MFCC = os.path.join(
    FOLDER_MODEL, "model_mfcc_terbaik.json"
)

FILE_MODEL_MEL = os.path.join(
    FOLDER_MODEL, "model_mel_terbaik.json"
)

FILE_NORMALISASI_MFCC = os.path.join(
    FOLDER_PREPROCESSING, "mfcc_parameter_normalisasi.json"
)

FILE_NORMALISASI_MEL = os.path.join(
    FOLDER_PREPROCESSING, "mel_parameter_normalisasi.json"
)


# ==========================================
# 2. KONFIGURASI STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Music Instrument Classifier",
    page_icon="♫",
    layout="centered"
)


# ==========================================
# 3. CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #F4F5F2;
}

.block-container {
    max-width: 760px;
    padding-top: 7rem !important;
    padding-bottom: 1rem !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.55rem !important;
}

.header-title {
    color: #26372D !important;
    font-size: 27px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.header-subtitle {
    color: #6D756E !important;
    font-size: 13px !important;
    margin-top: 3px !important;
}

.section-title {
    color: #35443A !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    margin-bottom: 2px !important;
}

div[data-testid="stRadio"] > div {
    gap: 15px !important;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px solid #D8DED7;
    border-radius: 8px;
    padding: 5px 10px;
}

div[data-testid="stAudio"] {
    margin-top: -5px;
}

.stButton > button {
    height: 42px;
    border-radius: 7px;
    background-color: #435D4B;
    color: white;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #354B3C;
    color: white;
    border: none;
}

.result-panel {
    background-color: #E8EDE8;
    border: 1px solid #D7E0D7;
    border-radius: 8px;
    padding: 13px 16px;
}

.result-label {
    color: #68766B !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}

.result-text {
    color: #35473A !important;
    font-size: 14px !important;
    margin: 0 !important;
}

hr {
    margin: 0.5rem 0 !important;
    border-color: #DDE1DA !important;
}

.footer {
    color: #818981 !important;
    font-size: 11px !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. MEMBACA JSON
# ==========================================

def baca_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================
# 5. NORMALISASI FITUR
# ==========================================

def normalisasi_fitur(fitur, parameter):

    mean = parameter["mean"]
    std = parameter["std"]

    fitur_normal = []

    for i in range(len(fitur)):

        if std[i] == 0:
            nilai = 0.0
        else:
            nilai = (fitur[i] - mean[i]) / std[i]

        fitur_normal.append(nilai)

    return fitur_normal


# ==========================================
# 6. PREDIKSI ANN
# ==========================================

def prediksi_ann(fitur, model):

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    _, _, _, probabilitas = forward_propagation(
        fitur,
        W1,
        b1,
        W2,
        b2
    )

    label_mapping = model["label_mapping"]

    mapping_index = {
        index: nama
        for nama, index in label_mapping.items()
    }

    indeks_prediksi = probabilitas.index(
        max(probabilitas)
    )

    nama_prediksi = mapping_index[indeks_prediksi]

    return nama_prediksi, probabilitas, mapping_index


# ==========================================
# 7. KLASIFIKASI AUDIO
# ==========================================

def klasifikasi_audio(file_audio, metode):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as file_temp:

        file_temp.write(file_audio.getvalue())
        path_audio = file_temp.name

    try:

        # Ekstraksi fitur
        fitur_mfcc, fitur_mel = ekstraksi_fitur_audio(
            path_audio
        )

        # Pemilihan model
        if metode == "MFCC":

            fitur = fitur_mfcc

            model = baca_json(FILE_MODEL_MFCC)

            parameter_normalisasi = baca_json(
                FILE_NORMALISASI_MFCC
            )

        else:

            fitur = fitur_mel

            model = baca_json(FILE_MODEL_MEL)

            parameter_normalisasi = baca_json(
                FILE_NORMALISASI_MEL
            )

        # Normalisasi
        fitur_normal = normalisasi_fitur(
            fitur,
            parameter_normalisasi
        )

        # Prediksi
        nama_prediksi, probabilitas, mapping_index = prediksi_ann(
            fitur_normal,
            model
        )

        return nama_prediksi, probabilitas, mapping_index

    finally:

        if os.path.exists(path_audio):
            os.remove(path_audio)


# ==========================================
# 8. HEADER
# ==========================================

st.markdown("""
<div class="header-title">Music Instrument Classifier</div>
<div class="header-subtitle">
    Klasifikasi instrumen musik menggunakan Artificial Neural Network
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# 9. LAYOUT 2 KOLOM
# ==========================================

kolom_input, kolom_hasil = st.columns(
    [1.15, 1],
    gap="medium"
)


# ==========================================
# 10. KOLOM INPUT AUDIO
# ==========================================

with kolom_input:

    st.markdown(
        '<div class="section-title">Metode Ekstraksi Fitur</div>',
        unsafe_allow_html=True
    )

    metode = st.radio(
        "Metode",
        ["MFCC", "Mel-Spectrogram"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="section-title">Input Audio</div>',
        unsafe_allow_html=True
    )

    file_audio = st.file_uploader(
        "Pilih file audio WAV",
        type=["wav"],
        label_visibility="collapsed"
    )

    if file_audio is not None:

        st.audio(file_audio)

        st.caption(
            f"File dipilih: {file_audio.name}"
        )

    if st.button(
        "Klasifikasi Audio",
        use_container_width=True
    ):

        if file_audio is None:

            st.warning(
                "Silakan masukkan file audio terlebih dahulu."
            )

        else:

            try:

                with st.spinner(
                    "Audio sedang diklasifikasi..."
 
                ):

                    nama_prediksi, probabilitas, mapping_index = (
                        klasifikasi_audio(
                            file_audio,
                            metode
                        )
                    )

                st.session_state["hasil"] = {
                    "nama": nama_prediksi,
                    "probabilitas": probabilitas,
                    "mapping_index": mapping_index,
                    "metode": metode
                }

            except Exception as error:

                st.session_state.pop("hasil", None)

                st.error(
                    f"Terjadi kesalahan saat klasifikasi: {error}"
                )


# ==========================================
# 11. KOLOM HASIL PREDIKSI
# ==========================================

with kolom_hasil:

    st.markdown("""
    <div class="result-panel">
        <div class="result-label">HASIL PREDIKSI</div>
    </div>
    """, unsafe_allow_html=True)

    hasil = st.session_state.get("hasil")

    if hasil is not None:

        nama_instrumen = hasil["nama"].replace(
            "_", " "
        )

        st.success(
            f"Instrumen terdeteksi: **{nama_instrumen}**"
        )

        st.caption(
            f"Metode ekstraksi fitur: {hasil['metode']}"
        )

        st.markdown(
            '<div class="section-title">Probabilitas Prediksi</div>',
            unsafe_allow_html=True
        )

        for i, nilai in enumerate(
            hasil["probabilitas"]
        ):

            nama_kelas = hasil[
                "mapping_index"
            ][i].replace("_", " ")

            persentase = nilai * 100

            st.write(
                f"**{nama_kelas}** — {persentase:.2f}%"
            )

            st.progress(
                float(max(0.0, min(1.0, nilai)))
            )

    else:

        st.markdown("""
        <div class="result-panel">
            <div class="result-text">
                Hasil klasifikasi akan ditampilkan di sini.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    ANN &nbsp; | &nbsp; Acoustic Guitar · Bass Guitar · Banjo
</div>
""", unsafe_allow_html=True)