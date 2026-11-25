
# ==========================================
# KONFIGURASI DATASET
# ==========================================

import os

# Direktori utama proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Direktori dataset
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Direktori penyimpanan hasil
HASIL_DIR = os.path.join(BASE_DIR, "hasil")

# Kategori instrumen musik
KELAS = [
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Banjo"
]

# Jumlah data setiap kelas
JUMLAH_DATA = {
    "Acoustic_Guitar": 3654,
    "Bass_Guitar": 3613,
    "Banjo": 2998
}

# ==========================================
# KONFIGURASI AUDIO
# ==========================================

SAMPLE_RATE = 22050
FRAME_LENGTH = 1024
HOP_LENGTH = 256
JUMLAH_FILTER_MEL = 26

# ==========================================
# KONFIGURASI EKSTRAKSI FITUR
# ==========================================

JUMLAH_KOEFISIEN_MFCC = 13
VARIANCE_THRESHOLD = 0.001