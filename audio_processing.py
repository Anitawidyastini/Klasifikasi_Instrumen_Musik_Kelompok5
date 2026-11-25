
import wave
import struct
import os
import math

# ==========================================
# MEMBACA INFORMASI FILE AUDIO
# ==========================================

def baca_audio(path_audio):

    with wave.open(path_audio, "rb") as audio:

        jumlah_channel = audio.getnchannels()
        sample_rate = audio.getframerate()
        bit_depth = audio.getsampwidth() * 8
        jumlah_frame = audio.getnframes()
        durasi = jumlah_frame / sample_rate

        data_audio = audio.readframes(jumlah_frame)

    print("\nINFORMASI AUDIO")
    print("-" * 40)
    print("Nama file       :", os.path.basename(path_audio))
    print("Sample rate     :", sample_rate, "Hz")
    print("Jumlah channel  :", jumlah_channel)
    print("Bit depth      :", bit_depth, "bit")
    print("Jumlah sampel  :", jumlah_frame)
    print("Durasi         :", round(durasi, 3), "detik")

    # ==========================================
    # KONVERSI DATA AUDIO MENJADI NILAI SAMPEL
    # ==========================================

    if bit_depth == 16:

        jumlah_nilai = len(data_audio) // 2

        format_data = "<" + str(jumlah_nilai) + "h"

        sampel_audio = list(
            struct.unpack(format_data, data_audio)
        )

    else:

        print("\nBit depth belum didukung untuk konversi sampel.")
        sampel_audio = []

    return sampel_audio, sample_rate, jumlah_channel, bit_depth, durasi


# ==========================================
# FRAMING AUDIO
# ==========================================

def buat_frame(sampel_audio, frame_length=1024, hop_length=256):

    frames = []

    panjang_audio = len(sampel_audio)

    posisi_awal = 0

    while posisi_awal + frame_length <= panjang_audio:

        frame = sampel_audio[
            posisi_awal : posisi_awal + frame_length
        ]

        frames.append(frame)

        posisi_awal += hop_length

    return frames


# ==========================================
# HAMMING WINDOW
# ==========================================

def hamming_window(panjang_frame):

    window = []

    for n in range(panjang_frame):

        nilai = 0.54 - 0.46 * (
            math.cos(
                (2 * math.pi * n) / (panjang_frame - 1)
            )
        )

        window.append(nilai)

    return window


# ==========================================
# MENERAPKAN WINDOW PADA FRAME
# ==========================================

def terapkan_window(frames):

    hasil_windowing = []

    panjang_frame = len(frames[0])

    window = hamming_window(panjang_frame)

    for frame in frames:

        frame_window = []

        for n in range(panjang_frame):

            nilai = frame[n] * window[n]

            frame_window.append(nilai)

        hasil_windowing.append(frame_window)

    return hasil_windowing


# ==========================================
# DISCRETE FOURIER TRANSFORM (DFT)
# ==========================================
def hitung_dft(frame):

    N = len(frame)

    hasil_real = []
    hasil_imajiner = []
    magnitude = []

    for k in range(N // 2 + 1):

        real = 0.0
        imajiner = 0.0

        sudut = 2 * math.pi * k / N

        cos_delta = math.cos(sudut)
        sin_delta = math.sin(sudut)

        cos_n = 1.0
        sin_n = 0.0

        for n in range(N):

            real += frame[n] * cos_n
            imajiner -= frame[n] * sin_n

            cos_next = cos_n * cos_delta - sin_n * sin_delta
            sin_next = sin_n * cos_delta + cos_n * sin_delta

            cos_n = cos_next
            sin_n = sin_next

        hasil_real.append(real)
        hasil_imajiner.append(imajiner)

        nilai_magnitude = math.sqrt(
            real * real + imajiner * imajiner
        )

        magnitude.append(nilai_magnitude)

    return hasil_real, hasil_imajiner, magnitude


# ==========================================
# FAST FOURIER TRANSFORM (FFT) MANUAL
# ==========================================

def hitung_fft(frame):

    N = len(frame)

    data = [complex(nilai, 0) for nilai in frame]

    # ==========================================
    # BIT-REVERSAL PERMUTATION
    # ==========================================

    j = 0

    for i in range(1, N):

        bit = N // 2

        while j & bit:
            j ^= bit
            bit //= 2

        j ^= bit

        if i < j:
            data[i], data[j] = data[j], data[i]

    # ==========================================
    # FFT RADIX-2 (BUTTERFLY)
    # ==========================================

    ukuran = 2

    while ukuran <= N:

        setengah = ukuran // 2

        sudut = -2 * math.pi / ukuran

        faktor_cos = math.cos(sudut)
        faktor_sin = math.sin(sudut)

        for awal in range(0, N, ukuran):

            w = complex(1.0, 0.0)

            for k in range(setengah):

                indeks_atas = awal + k
                indeks_bawah = indeks_atas + setengah

                nilai_atas = data[indeks_atas]
                nilai_bawah = data[indeks_bawah] * w

                data[indeks_atas] = nilai_atas + nilai_bawah
                data[indeks_bawah] = nilai_atas - nilai_bawah

                w = complex(
                    w.real * faktor_cos - w.imag * faktor_sin,
                    w.real * faktor_sin + w.imag * faktor_cos
                )

        ukuran *= 2

    # ==========================================
    # MENGAMBIL 513 BIN FREKUENSI
    # ==========================================

    hasil_real = []
    hasil_imajiner = []
    magnitude = []

    for k in range(N // 2 + 1):

        real = data[k].real
        imajiner = data[k].imag

        hasil_real.append(real)
        hasil_imajiner.append(imajiner)

        nilai_magnitude = math.sqrt(
            real * real + imajiner * imajiner
        )

        magnitude.append(nilai_magnitude)

    return hasil_real, hasil_imajiner, magnitude

# ==========================================
# MEL FILTER BANK
# ==========================================

def hz_to_mel(frekuensi):
    return 2595 * math.log10(1 + frekuensi / 700)


def mel_to_hz(nilai_mel):
    return 700 * (10 ** (nilai_mel / 2595) - 1)


def buat_mel_filter_bank(
    sample_rate=22050,
    n_fft=1024,
    jumlah_filter=26
):

    frekuensi_min = 0
    frekuensi_max = sample_rate / 2

    mel_min = hz_to_mel(frekuensi_min)
    mel_max = hz_to_mel(frekuensi_max)

    # Membuat 28 titik Mel untuk 26 filter
    titik_mel = []

    for i in range(jumlah_filter + 2):

        nilai = mel_min + (
            (mel_max - mel_min) * i / (jumlah_filter + 1)
        )

        titik_mel.append(mel_to_hz(nilai))

    # Mengubah frekuensi menjadi indeks bin DFT
    titik_bin = []

    for frekuensi in titik_mel:

        indeks = int(
            frekuensi * n_fft / sample_rate
        )

        titik_bin.append(indeks)

    # Membentuk filter segitiga
    filter_bank = []

    jumlah_bin = n_fft // 2 + 1

    for m in range(1, jumlah_filter + 1):

        filter_segitiga = [0.0] * jumlah_bin

        bin_kiri = titik_bin[m - 1]
        bin_tengah = titik_bin[m]
        bin_kanan = titik_bin[m + 1]

        # Sisi naik segitiga
        for k in range(bin_kiri, bin_tengah):

            if bin_tengah != bin_kiri:

                filter_segitiga[k] = (
                    (k - bin_kiri) /
                    (bin_tengah - bin_kiri)
                )

        # Sisi turun segitiga
        for k in range(bin_tengah, bin_kanan):

            if bin_kanan != bin_tengah:

                filter_segitiga[k] = (
                    (bin_kanan - k) /
                    (bin_kanan - bin_tengah)
                )

        filter_bank.append(filter_segitiga)

    return filter_bank, titik_bin


# ==========================================
# LOG MEL ENERGY
# ==========================================

def hitung_log_mel(energi_mel):

    hasil_log = []

    epsilon = 1e-10

    for energi in energi_mel:

        nilai_log = math.log(energi + epsilon)

        hasil_log.append(nilai_log)

    return hasil_log

# ==========================================
# DISCRETE COSINE TRANSFORM (DCT)
# ==========================================

def hitung_dct(log_mel):

    jumlah_data = len(log_mel)
    hasil_dct = []

    for n in range(jumlah_data):

        nilai_dct = 0.0

        for m in range(jumlah_data):

            nilai_cos = math.cos(
                math.pi * n * (m + 0.5) / jumlah_data
            )

            nilai_dct += log_mel[m] * nilai_cos

        hasil_dct.append(nilai_dct)

    return hasil_dct

# ==========================================
# PEMILIHAN 13 KOEFISIEN MFCC
# ==========================================

def pilih_koefisien_mfcc(hasil_dct, jumlah_koefisien=13):

    koefisien_mfcc = hasil_dct[:jumlah_koefisien]

    return koefisien_mfcc

# ==========================================
# EKSTRAKSI MFCC SELURUH FRAME
# ==========================================

def ekstraksi_mfcc_seluruh_frame(frames_window, filter_bank):

    seluruh_mfcc = []

    for frame in frames_window:

        # DFT
        _, _, magnitude = hitung_dft(frame)

        # Mel Filter Bank
        energi_mel = []

        for filter_segitiga in filter_bank:

            energi = 0.0

            for k in range(len(magnitude)):

                energi += magnitude[k] * filter_segitiga[k]

            energi_mel.append(energi)

        # Log Mel Energy
        hasil_log_mel = hitung_log_mel(energi_mel)

        # DCT
        hasil_dct = hitung_dct(hasil_log_mel)

        # Pemilihan 13 koefisien MFCC
        hasil_mfcc = pilih_koefisien_mfcc(hasil_dct)

        seluruh_mfcc.append(hasil_mfcc)

    return seluruh_mfcc

# ==========================================
# MEAN DAN STANDAR DEVIASI MFCC
# ==========================================

def hitung_mean_std_mfcc(seluruh_mfcc):

    jumlah_frame = len(seluruh_mfcc)
    jumlah_koefisien = len(seluruh_mfcc[0])

    mean_mfcc = []
    std_mfcc = []

    for k in range(jumlah_koefisien):

        # Menghitung mean
        total = 0.0

        for frame in seluruh_mfcc:
            total += frame[k]

        mean = total / jumlah_frame
        mean_mfcc.append(mean)

        # Menghitung varians
        total_varians = 0.0

        for frame in seluruh_mfcc:
            selisih = frame[k] - mean
            total_varians += selisih * selisih

        varians = total_varians / jumlah_frame

        # Menghitung standar deviasi
        std = math.sqrt(varians)

        std_mfcc.append(std)

    # Menggabungkan mean dan standar deviasi
    fitur_mfcc = mean_mfcc + std_mfcc

    return mean_mfcc, std_mfcc, fitur_mfcc

# ==========================================
# EKSTRAKSI MEL-SPECTROGRAM SELURUH FRAME
# ==========================================

def ekstraksi_mel_spectrogram_seluruh_frame(frames_window, filter_bank):

    seluruh_mel_spectrogram = []

    for frame in frames_window:

        # Menghitung DFT
        hasil_real, hasil_imajiner, magnitude = hitung_dft(frame)

        # Menghitung energi Mel
        energi_mel = []

        for filter_segitiga in filter_bank:

            energi = 0.0

            for k in range(len(magnitude)):

                energi += magnitude[k] * filter_segitiga[k]

            energi_mel.append(energi)

        # Mengubah energi Mel menjadi Log Mel
        log_mel = hitung_log_mel(energi_mel)

        seluruh_mel_spectrogram.append(log_mel)

    return seluruh_mel_spectrogram

# ==========================================
# MEAN DAN STANDAR DEVIASI MEL-SPECTROGRAM
# ==========================================

def hitung_mean_std_mel_spectrogram(seluruh_mel_spectrogram):

    jumlah_frame = len(seluruh_mel_spectrogram)
    jumlah_filter = len(seluruh_mel_spectrogram[0])

    mean_mel = []
    std_mel = []

    for k in range(jumlah_filter):

        # Menghitung mean
        total = 0.0

        for frame in seluruh_mel_spectrogram:
            total += frame[k]

        mean = total / jumlah_frame
        mean_mel.append(mean)

        # Menghitung varians
        total_varians = 0.0

        for frame in seluruh_mel_spectrogram:
            selisih = frame[k] - mean
            total_varians += selisih * selisih

        varians = total_varians / jumlah_frame

        # Menghitung standar deviasi
        std = math.sqrt(varians)

        std_mel.append(std)

    # Menggabungkan mean dan standar deviasi
    fitur_mel_spectrogram = mean_mel + std_mel

    return mean_mel, std_mel, fitur_mel_spectrogram

# ==========================================
# EKSTRAKSI MFCC DAN MEL-SPECTROGRAM BERSAMA
# ==========================================

def ekstrasi_mfcc_dan_mel_seluruh_frame(
    frames_window,
    filter_bank
):
    seluruh_mfcc = []
    seluruh_mel_spectrogram = []

    for frame in frames_window:

        # Menghitung FFT satu kali
        _, _, magnitude = hitung_fft(frame)

        # Menghitung energi Mel
        energi_mel = []

        for filter_segitiga in filter_bank:

            energi = 0.0

            for k in range(len(magnitude)):

                energi += magnitude[k] * filter_segitiga[k]

            energi_mel.append(energi)

        # Mengubah energi Mel menjadi Log Mel
        log_mel = hitung_log_mel(energi_mel)

        # ==========================================
        # EKSTRAKSI MFCC
        # ==========================================

        hasil_dct = hitung_dct(log_mel)

        hasil_mfcc = pilih_koefisien_mfcc(
            hasil_dct
        )

        seluruh_mfcc.append(hasil_mfcc)

        # ==========================================
        # EKSTRAKSI MEL-SPECTROGRAM
        # ==========================================

        seluruh_mel_spectrogram.append(log_mel)

    return seluruh_mfcc, seluruh_mel_spectrogram