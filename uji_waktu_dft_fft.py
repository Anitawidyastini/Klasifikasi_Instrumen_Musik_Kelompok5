
import time

from dataset_loader import ambil_daftar_audio
from audio_processing import (
    baca_audio,
    buat_frame,
    hamming_window,
    terapkan_window,
    hitung_dft,
    hitung_fft
)

# Mengambil satu audio untuk pengujian
daftar_audio = ambil_daftar_audio()
path_audio, label = daftar_audio[0]

sampel_audio, sample_rate, kanal, bit_depth, durasi = baca_audio(path_audio)

frames = buat_frame(sampel_audio)
frames_window = terapkan_window(frames)

# Mengambil 20 frame non-silent yang sama untuk kedua algoritma
frames_uji = [
    frame for frame in frames_window
    if any(nilai != 0 for nilai in frame)
][:20]

print("=" * 50)
print("PENGUJIAN WAKTU DFT DAN FFT")
print("=" * 50)
print(f"File audio : {path_audio}")
print(f"Label     : {label}")
print(f"Jumlah frame yang diuji : {len(frames_uji)}")
print()

# Warm-up kedua algoritma (tidak masuk pengukuran)
if frames_uji:
    hitung_dft(frames_uji[0])
    hitung_fft(frames_uji[0])

# Pengujian waktu DFT
print("Mengukur waktu DFT...")
mulai_dft = time.perf_counter()

for frame in frames_uji:
    hitung_dft(frame)

waktu_dft = time.perf_counter() - mulai_dft

# Pengujian waktu FFT
print("Mengukur waktu FFT...")
mulai_fft = time.perf_counter()

for frame in frames_uji:
    hitung_fft(frame)

waktu_fft = time.perf_counter() - mulai_fft

# Menampilkan hasil
print()
print("=" * 50)
print("HASIL PENGUJIAN")
print("=" * 50)

print(f"Waktu DFT : {waktu_dft:.6f} detik")
print(f"Waktu FFT : {waktu_fft:.6f} detik")

if waktu_fft > 0:
    percepatan = waktu_dft / waktu_fft
    print(f"Percepatan FFT : {percepatan:.2f} kali")
    
    penghematan = ((waktu_dft - waktu_fft) / waktu_dft) * 100
    print(f"Penghematan waktu : {penghematan:.2f}%")

print("=" * 50)