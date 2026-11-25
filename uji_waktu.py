import time
import os

from dataset_loader import ambil_daftar_audio
from feature_extraction import ekstraksi_fitur_audio


# ==========================================
# PENGUJIAN WAKTU EKSTRAKSI
# ==========================================

daftar_audio = ambil_daftar_audio()

# Menggunakan audio kedua agar tidak mengulang
# audio 1.wav yang sudah pernah diuji
path_audio, label = daftar_audio[1]

nama_file = os.path.basename(path_audio)

print("=" * 50)
print("PENGUJIAN WAKTU EKSTRAKSI FITUR")
print("=" * 50)
print("Nama audio :", nama_file)
print("Label     :", label)
print("Memulai proses ekstraksi...")
print("=" * 50)

waktu_mulai = time.time()

fitur_mfcc, fitur_mel = ekstraksi_fitur_audio(path_audio)

waktu_selesai = time.time()

total_waktu = waktu_selesai - waktu_mulai

print("\nEKSTRAKSI BERHASIL")
print("Jumlah fitur MFCC :", len(fitur_mfcc))
print("Jumlah fitur Mel   :", len(fitur_mel))
print("Waktu proses      :", round(total_waktu, 2), "detik")
print("=" * 50)