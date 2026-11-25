
# ==========================================
# EKSTRAKSI FITUR SELURUH DATASET
# ==========================================

import os
import csv

from dataset_loader import ambil_daftar_audio
from feature_extraction import ekstraksi_fitur_audio
from config import HASIL_DIR


# ==========================================
# KONFIGURASI EKSTRAKSI
# ==========================================

MAX_AUDIO = None


# ==========================================
# PENYIMPANAN DATA FITUR
# ==========================================

def siapkan_file_csv(path_csv, nama_fitur, jumlah_fitur):

    header = ["nama_file", "label"]

    for i in range(jumlah_fitur):
        header.append(f"{nama_fitur}_{i + 1}")

    if not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0:

        with open(path_csv, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)
            writer.writerow(header)


def baca_data_selesai(path_csv):

    data_selesai = set()

    if not os.path.exists(path_csv):
        return data_selesai

    with open(path_csv, "r", newline="", encoding="utf-8") as file:

        reader = csv.reader(file)
        next(reader, None)

        for baris in reader:

            if len(baris) >= 2:
                data_selesai.add((baris[0], baris[1]))

    return data_selesai


def simpan_satu_data(path_csv, nama_file, label, fitur):

    with open(path_csv, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        writer.writerow([nama_file, label] + fitur)


# ==========================================
# PROSES EKSTRAKSI DATASET
# ==========================================

def ekstraksi_seluruh_dataset():

    os.makedirs(HASIL_DIR, exist_ok=True)

    path_csv_mfcc = os.path.join(
        HASIL_DIR,
        "fitur_mfcc_dataset.csv"
    )

    path_csv_mel = os.path.join(
        HASIL_DIR,
        "fitur_mel_dataset.csv"
    )

    siapkan_file_csv(path_csv_mfcc, "mfcc", 26)
    siapkan_file_csv(path_csv_mel, "mel", 52)

    selesai_mfcc = baca_data_selesai(path_csv_mfcc)
    selesai_mel = baca_data_selesai(path_csv_mel)

    daftar_audio = ambil_daftar_audio()

    if MAX_AUDIO is not None:
        daftar_audio = daftar_audio[:MAX_AUDIO]

    total_audio = len(daftar_audio)

    print("=" * 50)
    print("EKSTRAKSI FITUR DATASET")
    print("=" * 50)
    print("Jumlah audio yang diproses :", total_audio)
    print("Fitur MFCC                : 26")
    print("Fitur Mel-Spectrogram     : 52")
    print("=" * 50)

    for indeks, (path_audio, label) in enumerate(daftar_audio, start=1):

        nama_file = os.path.basename(path_audio)
        identitas_data = (nama_file, label)

        if (
            identitas_data in selesai_mfcc
            and identitas_data in selesai_mel
        ):
            print(
                f"[{indeks}/{total_audio}] "
                f"{nama_file} - {label} - sudah tersimpan"
            )
            continue

        print(
            f"\n[{indeks}/{total_audio}] "
            f"Memproses {nama_file} - {label}"
        )

        fitur_mfcc, fitur_mel = ekstraksi_fitur_audio(path_audio)

        if identitas_data not in selesai_mfcc:

            simpan_satu_data(
                path_csv_mfcc,
                nama_file,
                label,
                fitur_mfcc
            )

            selesai_mfcc.add(identitas_data)

        if identitas_data not in selesai_mel:

            simpan_satu_data(
                path_csv_mel,
                nama_file,
                label,
                fitur_mel
            )

            selesai_mel.add(identitas_data)

        print("MFCC berhasil disimpan :", len(fitur_mfcc), "fitur")
        print("Mel berhasil disimpan   :", len(fitur_mel), "fitur")

    print("\n" + "=" * 50)
    print("EKSTRAKSI SELESAI")
    print("=" * 50)
    print("Data MFCC tersimpan :", len(selesai_mfcc))
    print("Data Mel tersimpan   :", len(selesai_mel))
    print("Lokasi hasil        :", HASIL_DIR)


# ==========================================
# PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    ekstraksi_seluruh_dataset()