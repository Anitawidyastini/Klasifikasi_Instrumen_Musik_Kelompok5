
import os
from audio_processing import (
    baca_audio,
    buat_frame,
    terapkan_window,
    hitung_dft,
    buat_mel_filter_bank,
    hitung_log_mel,
    hitung_dct,
    pilih_koefisien_mfcc,
    ekstraksi_mfcc_seluruh_frame,
    hitung_mean_std_mfcc,
    ekstraksi_mel_spectrogram_seluruh_frame,
    hitung_mean_std_mel_spectrogram
)
from dataset_loader import ambil_daftar_audio
from feature_extraction import ekstraksi_fitur_audio 
from csv_writer import simpan_fitur_csv
from config import HASIL_DIR

# ==========================================
# KONFIGURASI DATASET
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

KELAS_DATASET = [
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Banjo"
]

TARGET_DATASET = {
    "Acoustic_Guitar": 3654,
    "Bass_Guitar": 3613,
    "Banjo": 2998
}


# ==========================================
# PEMERIKSAAN DATASET
# ==========================================

def periksa_dataset():

    print("=" * 50)
    print("PEMERIKSAAN DATASET INSTRUMEN MUSIK")
    print("=" * 50)

    total_file = 0

    for kelas in KELAS_DATASET:

        folder_kelas = os.path.join(DATASET_DIR, kelas)

        print("\nKelas:", kelas)

        if not os.path.exists(folder_kelas):
            print("Folder tidak ditemukan")
            continue

        file_audio = []

        for nama_file in os.listdir(folder_kelas):

            if nama_file.lower().endswith(".wav"):
                file_audio.append(nama_file)

        jumlah_file = len(file_audio)
        total_file += jumlah_file

        target = TARGET_DATASET[kelas]

        print("Jumlah file WAV :", jumlah_file)
        print("Target awal     :", target)

        if jumlah_file == target:
            print("Status         : Sesuai target")
        else:
            print("Status         : Jumlah berbeda dari target")

    print("\n" + "=" * 50)
    print("TOTAL FILE AUDIO :", total_file)
    print("TARGET TOTAL    :", 10265)
    print("=" * 50)


# ==========================================
# PENGUJIAN PEMBACAAN AUDIO
# ==========================================

def uji_baca_audio():

    folder_audio = os.path.join(
        DATASET_DIR,
        "Acoustic_Guitar"
    )

    file_audio = [
        nama for nama in os.listdir(folder_audio)
        if nama.lower().endswith(".wav")
    ]

    if not file_audio:
        print("File audio tidak ditemukan.")
        return

    path_audio = os.path.join(
        folder_audio,
        file_audio[0]
    )

    sampel, sample_rate, channel, bit_depth, durasi = baca_audio(
        path_audio
    )

    print("\nHASIL PEMBACAAN SAMPEL AUDIO")
    print("-" * 40)
    print("Jumlah nilai sampel :", len(sampel))
    print("5 sampel pertama   :", sampel[:5])

    # ==========================================
    # PEMERIKSAAN NILAI AMPLITUDO AUDIO
    # ==========================================

    nilai_maksimum = max(sampel)
    nilai_minimum = min(sampel)

    jumlah_nol = 0

    for nilai in sampel:
        if nilai == 0:
            jumlah_nol += 1

    print("\nPEMERIKSAAN AMPLITUDO AUDIO")
    print("-" * 40)
    print("Nilai maksimum         :", nilai_maksimum)
    print("Nilai minimum         :", nilai_minimum)
    print("Jumlah sampel nol     :", jumlah_nol)
    print("Persentase sampel nol:", round(
        (jumlah_nol / len(sampel)) * 100, 2
    ), "%")

    print("\nSAMPEL PADA POSISI BERBEDA")
    print("-" * 40)

    posisi = [0, 100, 1000, 5000, 10000, 30000, 60000]

    for indeks in posisi:
        print(f"Sampel ke-{indeks} :", sampel[indeks])

    # ==========================================
    # PENGUJIAN FRAMING
    # ==========================================

    frames = buat_frame(sampel)

    print("\nHASIL FRAMING")
    print("-" * 40)
    print("Panjang frame       :", len(frames[0]))
    print("Hop length         :", 256)
    print("Overlap            :", 768)
    print("Jumlah frame       :", len(frames))
    print("5 sampel frame 1  :", frames[0][:5])
    print("5 sampel frame 2  :", frames[1][:5])

    # ==========================================
    # PENGUJIAN HAMMING WINDOW
    # ==========================================

    frames_window = terapkan_window(frames)

    print("\nHASIL HAMMING WINDOW")
    print("-" * 40)
    print("Jumlah frame          :", len(frames_window))
    print("Panjang window       :", len(frames_window[0]))
    print("5 sampel sebelum    :", frames[10][:5])
    print("5 sampel sesudah   :", frames_window[10][:5])

    # ==========================================
    # PENGUJIAN DISCRETE FOURIER TRANSFORM
    # ==========================================

    # Memilih frame dengan energi tertinggi untuk pengujian
    frame_uji = max(
        frames_window,
        key=lambda frame: sum(nilai * nilai for nilai in frame)
    )

    hasil_real, hasil_imajiner, magnitude = hitung_dft(frame_uji)

    print("\nHASIL DISCRETE FOURIER TRANSFORM")
    print("-" * 40)
    print("Panjang frame DFT :", len(frame_uji))
    print("Jumlah frekuensi  :", len(magnitude))
    print("5 nilai real      :", hasil_real[:5])
    print("5 nilai imajiner  :", hasil_imajiner[:5])
    print("5 magnitude awal  :", magnitude[:5])
    print("Magnitude maksimum :", max(magnitude))


    # ==========================================
    # PENGUJIAN MEL FILTER BANK
    # ==========================================

    filter_bank, titik_bin = buat_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=1024,
        jumlah_filter=26
    )

    energi_mel = []

    for filter_segitiga in filter_bank:

        energi = 0.0

        for k in range(len(magnitude)):

            energi += magnitude[k] * filter_segitiga[k]

        energi_mel.append(energi)

    print("\nHASIL MEL FILTER BANK")
    print("-" * 40)
    print("Jumlah filter       :", len(filter_bank))
    print("Jumlah bin DFT     :", len(magnitude))
    print("Jumlah titik batas  :", len(titik_bin))
    print("5 titik bin awal  :", titik_bin[:5])
    print("5 energi Mel awal :", energi_mel[:5])
    print("Energi maksimum    :", max(energi_mel))


    # ==========================================
    # PENGUJIAN LOG MEL ENERGY
    # ==========================================

    hasil_log_mel = hitung_log_mel(energi_mel)

    print("\nHASIL LOG MEL ENERGY")
    print("-" * 40)
    print("Jumlah nilai       :", len(hasil_log_mel))
    print("5 nilai log awal  :", hasil_log_mel[:5])
    print("Nilai log minimum :", min(hasil_log_mel))
    print("Nilai log maksimum:", max(hasil_log_mel))

    # ==========================================
    # PENGUJIAN DISCRETE COSINE TRANSFORM (DCT)
    # ==========================================

    hasil_dct = hitung_dct(hasil_log_mel)

    print("\nHASIL DISCRETE COSINE TRANSFORM (DCT)")
    print("-" * 40)
    print("Jumlah koefisien :", len(hasil_dct))
    print("13 koefisien awal:", hasil_dct[:13])
    print("Koefisien minimum :", min(hasil_dct))
    print("Koefisien maksimum :", max(hasil_dct))

    # ==========================================
    # PENGUJIAN PEMILIHAN KOEFISIEN MFCC
    # ==========================================

    hasil_mfcc = pilih_koefisien_mfcc(hasil_dct)

    print("\nHASIL MFCC")
    print("-" * 40)
    print("Jumlah koefisien :", len(hasil_mfcc))
    print("Nilai MFCC       :", hasil_mfcc)
    print("MFCC minimum     :", min(hasil_mfcc))
    print("MFCC maksimum    :", max(hasil_mfcc))

    # ==========================================
    # PENGUJIAN MFCC SELURUH FRAME
    # ==========================================

    seluruh_mfcc = ekstraksi_mfcc_seluruh_frame(
        frames_window,
        filter_bank
    )

    print("\nHASIL MFCC SELURUH FRAME")
    print("-" * 40)
    print("Jumlah frame MFCC :", len(seluruh_mfcc))
    print("Koefisien per frame :", len(seluruh_mfcc[0]))
    print("MFCC frame pertama :", seluruh_mfcc[0])
    print("MFCC frame terakhir :", seluruh_mfcc[-1])

    # ==========================================
    # PENGUJIAN MEAN DAN STANDAR DEVIASI MFCC
    # ==========================================

    mean_mfcc, std_mfcc, fitur_mfcc = hitung_mean_std_mfcc(
        seluruh_mfcc
    )

    print("\nHASIL MEAN DAN STANDAR DEVIASI MFCC")
    print("-" * 40)
    print("Jumlah mean        :", len(mean_mfcc))
    print("Jumlah std         :", len(std_mfcc))
    print("Jumlah fitur akhir :", len(fitur_mfcc))
    print("13 nilai mean     :", mean_mfcc)
    print("13 nilai std      :", std_mfcc)
    print("26 fitur MFCC    :", fitur_mfcc)

    # ==========================================
    # PENGUJIAN MEL-SPECTROGRAM SELURUH FRAME
    # ==========================================

    seluruh_mel_spectrogram = ekstraksi_mel_spectrogram_seluruh_frame(
        frames_window,
        filter_bank
    )

    print("\nHASIL MEL-SPECTROGRAM")
    print("-" * 40)
    print("Jumlah frame          :", len(seluruh_mel_spectrogram))
    print("Jumlah filter per frame :", len(seluruh_mel_spectrogram[0]))
    print("Mel frame pertama    :", seluruh_mel_spectrogram[0])
    print("Mel frame terakhir   :", seluruh_mel_spectrogram[-1])

    # ==========================================
    # PENGUJIAN MEAN DAN STANDAR DEVIASI MEL-SPECTROGRAM
    # ==========================================

    mean_mel, std_mel, fitur_mel_spectrogram = hitung_mean_std_mel_spectrogram(
        seluruh_mel_spectrogram
    )

    print("\nHASIL MEAN DAN STANDAR DEVIASI MEL-SPECTROGRAM")
    print("-" * 40)
    print("Jumlah mean        :", len(mean_mel))
    print("Jumlah std         :", len(std_mel))
    print("Jumlah fitur akhir :", len(fitur_mel_spectrogram))
    print("26 nilai mean     :", mean_mel)
    print("26 nilai std      :", std_mel)
    print("52 fitur Mel     :", fitur_mel_spectrogram)

    # ==========================================
    # PENGUJIAN PEMBACA DATASET
    # ==========================================

    daftar_audio = ambil_daftar_audio()

    print("\nHASIL PEMBACAAN DATASET")
    print("-" * 40)

    print("Total file audio :", len(daftar_audio))

    jumlah_per_kelas = {}

    for path_audio, kelas in daftar_audio:

        if kelas not in jumlah_per_kelas:
            jumlah_per_kelas[kelas] = 0

        jumlah_per_kelas[kelas] += 1

    print("\nJumlah audio per kelas:")

    for kelas, jumlah in jumlah_per_kelas.items():
        print(kelas, ":", jumlah)

    print("\n5 data pertama:")

    for data in daftar_audio[:5]:
        print(data)

    print("\n5 data terakhir:")

    for data in daftar_audio[-5:]:
        print(data)

    # ==========================================
    # PENGUJIAN EKSTRAKSI FITUR GABUNGAN
    # ==========================================

    path_audio_uji = os.path.join(
        DATASET_DIR,
        "Acoustic_Guitar",
        "1.wav"
    )

    fitur_mfcc_uji, fitur_mel_uji = ekstraksi_fitur_audio(
        path_audio_uji
    )

    print("\nHASIL EKSTRAKSI FITUR GABUNGAN")
    print("-" * 40)

    print("Jumlah fitur MFCC :", len(fitur_mfcc_uji))
    print("Jumlah fitur Mel   :", len(fitur_mel_uji))

    print("\n5 fitur awal MFCC:")
    print(fitur_mfcc_uji[:5])

    print("\n5 fitur awal Mel-Spectrogram:")
    print(fitur_mel_uji[:5])


    # ==========================================
    # PENGUJIAN PENYIMPANAN FITUR KE CSV
    # ==========================================

    data_mfcc = [
        ["1.wav", "Acoustic_Guitar"] + fitur_mfcc_uji
    ]

    data_mel = [
        ["1.wav", "Acoustic_Guitar"] + fitur_mel_uji
    ]

    path_csv_mfcc = os.path.join(HASIL_DIR, "uji_fitur_mfcc.csv")
    path_csv_mel = os.path.join(HASIL_DIR, "uji_fitur_mel.csv")

    simpan_fitur_csv(
        path_csv_mfcc,
        data_mfcc,
        "mfcc"
    )

    simpan_fitur_csv(
        path_csv_mel,
        data_mel,
        "mel"
    )

    print("\nPENGUJIAN PENYIMPANAN CSV")
    print("-" * 40)
    print("CSV MFCC berhasil disimpan :", path_csv_mfcc)
    print("CSV Mel berhasil disimpan  :", path_csv_mel)

# ==========================================
# PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    periksa_dataset()

    print("\n\nMELAKUKAN UJI BACA AUDIO")
    uji_baca_audio()