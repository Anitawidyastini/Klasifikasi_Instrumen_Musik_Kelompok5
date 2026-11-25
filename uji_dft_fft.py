
import math
from dataset_loader import ambil_daftar_audio
from audio_processing import (
    baca_audio,
    buat_frame,
    terapkan_window,
    hitung_dft,
    hitung_fft
)

# ==========================================
# MENGAMBIL SATU AUDIO UNTUK PENGUJIAN
# ==========================================

daftar_audio = ambil_daftar_audio()

path_audio, label = daftar_audio[0]

sampel_audio, sample_rate, _, _, _ = baca_audio(path_audio)
print("\nPEMERIKSAAN DATA AUDIO")
print("-" * 40)

print("Jumlah sampel audio :", len(sampel_audio))
print("Nilai minimum audio :", min(sampel_audio))
print("Nilai maksimum audio :", max(sampel_audio))
print("10 sampel pertama   :", sampel_audio[:10])

frames = buat_frame(sampel_audio)
print("\nPEMERIKSAAN FRAMING")
print("-" * 40)

print("Jumlah frame :", len(frames))
print("Frame pertama:", frames[0][:10])

frames_window = terapkan_window(frames)

# ==========================================
# MEMILIH FRAME YANG MEMILIKI SINYAL
# ==========================================

indeks_frame = 0

for i, frame_sementara in enumerate(frames_window):

    if any(nilai != 0 for nilai in frame_sementara):

        indeks_frame = i
        break

frame = frames_window[indeks_frame]

print("\nPEMILIHAN FRAME")
print("-" * 40)
print("Indeks frame :", indeks_frame)
print("Nomor frame  :", indeks_frame + 1)
print("Min frame    :", min(frame))
print("Max frame    :", max(frame))

# ==========================================
# MEMERIKSA ISI FRAME
# ==========================================

print("\nPEMERIKSAAN SINYAL FRAME")
print("-" * 40)

print("Nilai minimum frame :", min(frame))
print("Nilai maksimum frame :", max(frame))

jumlah_nol = sum(1 for nilai in frame if nilai == 0)
jumlah_nonzero = sum(1 for nilai in frame if nilai != 0)

print("Jumlah sampel nol     :", jumlah_nol)
print("Jumlah sampel nonzero :", jumlah_nonzero)

print("\nAUDIO YANG DIUJI")
print("-" * 40)
print("Nama file :", path_audio)
print("Label     :", label)
print("Frame     :", 1)
print("Panjang   :", len(frame))


# ==========================================
# PENGUJIAN DFT
# ==========================================

print("\nMenghitung DFT...")

real_dft, imag_dft, mag_dft = hitung_dft(frame)


# ==========================================
# PENGUJIAN FFT
# ==========================================

print("Menghitung FFT...")

real_fft, imag_fft, mag_fft = hitung_fft(frame)

print("\nPEMERIKSAAN MAGNITUDE")
print("-" * 40)

print("Magnitude maksimum DFT :", max(mag_dft))
print("Magnitude maksimum FFT  :", max(mag_fft))

jumlah_magnitude_dft = sum(
    1 for nilai in mag_dft if nilai > 1e-10
)

jumlah_magnitude_fft = sum(
    1 for nilai in mag_fft if nilai > 1e-10
)

print("Bin aktif DFT :", jumlah_magnitude_dft)
print("Bin aktif FFT  :", jumlah_magnitude_fft)

# ==========================================
# MEMBANDINGKAN HASIL
# ==========================================

selisih_real = []
selisih_imag = []
selisih_magnitude = []

for i in range(len(mag_dft)):

    selisih_real.append(
        abs(real_dft[i] - real_fft[i])
    )

    selisih_imag.append(
        abs(imag_dft[i] - imag_fft[i])
    )

    selisih_magnitude.append(
        abs(mag_dft[i] - mag_fft[i])
    )


# ==========================================
# HASIL PENGUJIAN
# ==========================================

print("\nHASIL PERBANDINGAN DFT DAN FFT")
print("-" * 40)

print("Jumlah bin :", len(mag_dft))

print(
    "Selisih real maksimum      :",
    max(selisih_real)
)

print(
    "Selisih imajiner maksimum  :",
    max(selisih_imag)
)

print(
    "Selisih magnitude maksimum  :",
    max(selisih_magnitude)
)

print(
    "Magnitude DFT bin pertama  :",
    mag_dft[0]
)

print(
    "Magnitude FFT bin pertama  :",
    mag_fft[0]
)

print(
    "Magnitude DFT bin ke-100  :",
    mag_dft[100]
)

print(
    "Magnitude FFT bin ke-100  :",
    mag_fft[100]
)

print(
    "Magnitude DFT bin terakhir :",
    mag_dft[-1]
)

print(
    "Magnitude FFT bin terakhir :",
    mag_fft[-1]
)