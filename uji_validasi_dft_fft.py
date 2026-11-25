
from dataset_loader import ambil_daftar_audio
from audio_processing import (
    baca_audio,
    buat_frame,
    terapkan_window,
    hitung_dft,
    hitung_fft
)

# Mengambil satu audio
daftar_audio = ambil_daftar_audio()
path_audio, label = daftar_audio[0]

sampel_audio, sample_rate, kanal, bit_depth, durasi = baca_audio(path_audio)
frames = buat_frame(sampel_audio)
frames_window = terapkan_window(frames)

# Mengambil 20 frame non-silent
frames_uji = [
    frame for frame in frames_window
    if any(nilai != 0 for nilai in frame)
][:20]

# Batas toleransi perbedaan numerik
toleransi = 1e-9

selisih_real_terbesar = 0
selisih_imajiner_terbesar = 0
selisih_magnitude_terbesar = 0

jumlah_frame_sesuai = 0

print("=" * 50)
print("VALIDASI HASIL DFT DAN FFT")
print("=" * 50)
print(f"Audio : {path_audio}")
print(f"Label : {label}")
print(f"Frame diuji : {len(frames_uji)}")
print()

for nomor, frame in enumerate(frames_uji, start=1):

    # Menghitung DFT
    real_dft, imajiner_dft, magnitude_dft = hitung_dft(frame)

    # Menghitung FFT
    real_fft, imajiner_fft, magnitude_fft = hitung_fft(frame)

    # Menghitung selisih setiap komponen
    selisih_real = max(
        abs(a - b) for a, b in zip(real_dft, real_fft)
    )

    selisih_imajiner = max(
        abs(a - b) for a, b in zip(imajiner_dft, imajiner_fft)
    )

    selisih_magnitude = max(
        abs(a - b) for a, b in zip(magnitude_dft, magnitude_fft)
    )

    # Menyimpan selisih terbesar dari seluruh frame
    selisih_real_terbesar = max(
        selisih_real_terbesar, selisih_real
    )

    selisih_imajiner_terbesar = max(
        selisih_imajiner_terbesar, selisih_imajiner
    )

    selisih_magnitude_terbesar = max(
        selisih_magnitude_terbesar, selisih_magnitude
    )

    # Memeriksa kesesuaian hasil dalam batas toleransi
    if (
        selisih_real <= toleransi
        and selisih_imajiner <= toleransi
        and selisih_magnitude <= toleransi
    ):
        jumlah_frame_sesuai += 1

    print(
        f"Frame {nomor:2d} | "
        f"Real: {selisih_real:.3e} | "
        f"Imaginer: {selisih_imajiner:.3e} | "
        f"Magnitude: {selisih_magnitude:.3e}"
    )

print()
print("=" * 50)
print("HASIL VALIDASI")
print("=" * 50)

print(f"Frame sesuai toleransi : {jumlah_frame_sesuai}/{len(frames_uji)}")
print(f"Selisih real terbesar : {selisih_real_terbesar:.3e}")
print(f"Selisih imaginer terbesar : {selisih_imajiner_terbesar:.3e}")
print(f"Selisih magnitude terbesar : {selisih_magnitude_terbesar:.3e}")
print(f"Toleransi : {toleransi}")

if jumlah_frame_sesuai == len(frames_uji):
    print("\nHASIL: DFT DAN FFT KONSISTEN")
else:
    print("\nHASIL: TERDAPAT PERBEDAAN, PERLU DIPERIKSA")

print("=" * 50)