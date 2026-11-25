
import csv
import os
import random
import math
import json

from config import HASIL_DIR, VARIANCE_THRESHOLD


# ==========================================
# 1. KONFIGURASI
# ==========================================

RANDOM_SEED = 42

INPUT_MFCC = os.path.join(
    HASIL_DIR,
    "fitur_mfcc_dataset.csv"
)

INPUT_MEL = os.path.join(
    HASIL_DIR,
    "fitur_mel_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    HASIL_DIR,
    "preprocessing"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. MEMBACA DATA CSV
# ==========================================

def baca_csv(path):

    data = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        kolom_fitur = [
            kolom for kolom in reader.fieldnames
            if kolom not in ["nama_file", "label"]
        ]

        for baris in reader:

            data.append({
                "nama_file": baris["nama_file"],
                "label": baris["label"],
                "fitur": [
                    float(baris[kolom])
                    for kolom in kolom_fitur
                ]
            })

    return data, kolom_fitur


# ==========================================
# 3. MEMBAGI DATASET (STRATIFIED SPLIT)
# ==========================================

def bagi_dataset(data):

    random.seed(RANDOM_SEED)

    kelompok = {}

    for baris in data:

        label = baris["label"]

        if label not in kelompok:
            kelompok[label] = []

        kelompok[label].append(baris)

    train = []
    validasi = []
    test = []

    for label, data_kelas in kelompok.items():

        random.shuffle(data_kelas)

        jumlah = len(data_kelas)

        jumlah_train = int(jumlah * 0.70)
        jumlah_validasi = int(jumlah * 0.15)

        data_train = data_kelas[:jumlah_train]

        data_validasi = data_kelas[
            jumlah_train:
            jumlah_train + jumlah_validasi
        ]

        data_test = data_kelas[
            jumlah_train + jumlah_validasi:
        ]

        train.extend(data_train)
        validasi.extend(data_validasi)
        test.extend(data_test)

    random.shuffle(train)
    random.shuffle(validasi)
    random.shuffle(test)

    return train, validasi, test


# ==========================================
# 4. MENGHITUNG VARIANSI FITUR
# ==========================================

def hitung_variansi(data, indeks_fitur):

    nilai = [
        baris["fitur"][indeks_fitur]
        for baris in data
    ]

    mean = sum(nilai) / len(nilai)

    variansi = sum(
        (x - mean) ** 2
        for x in nilai
    ) / len(nilai)

    return variansi


# ==========================================
# 5. FEATURE REDUCTION
# ==========================================

def seleksi_fitur(
    train,
    validasi,
    test,
    nama_fitur
):

    fitur_dipertahankan = []
    fitur_dihapus = []

    for i, nama in enumerate(nama_fitur):

        variansi = hitung_variansi(train, i)

        if variansi >= VARIANCE_THRESHOLD:

            fitur_dipertahankan.append(i)

        else:

            fitur_dihapus.append(i)

    # Menghapus fitur yang tidak lolos threshold
    for dataset in [train, validasi, test]:

        for baris in dataset:

            baris["fitur"] = [
                baris["fitur"][i]
                for i in fitur_dipertahankan
            ]

    nama_fitur_terpilih = [
        nama_fitur[i]
        for i in fitur_dipertahankan
    ]

    return (
        fitur_dipertahankan,
        fitur_dihapus,
        nama_fitur_terpilih
    )


# ==========================================
# 6. NORMALISASI Z-SCORE
# ==========================================

def normalisasi_zscore(
    train,
    validasi,
    test
):

    jumlah_fitur = len(train[0]["fitur"])

    mean_fitur = []
    std_fitur = []

    # Mean dan standar deviasi dari training
    for i in range(jumlah_fitur):

        nilai = [
            baris["fitur"][i]
            for baris in train
        ]

        mean = sum(nilai) / len(nilai)

        variansi = sum(
            (x - mean) ** 2
            for x in nilai
        ) / len(nilai)

        std = math.sqrt(variansi)

        mean_fitur.append(mean)
        std_fitur.append(std)

    # Normalisasi seluruh dataset menggunakan parameter training
    for dataset in [train, validasi, test]:

        for baris in dataset:

            fitur_normalisasi = []

            for i, nilai in enumerate(baris["fitur"]):

                if std_fitur[i] == 0:

                    hasil = 0

                else:

                    hasil = (
                        nilai - mean_fitur[i]
                    ) / std_fitur[i]

                fitur_normalisasi.append(hasil)

            baris["fitur"] = fitur_normalisasi

    return mean_fitur, std_fitur


# ==========================================
# 7. MENYIMPAN PARAMETER NORMALISASI
# ==========================================

def simpan_parameter_normalisasi(
    mean_fitur,
    std_fitur,
    nama_fitur,
    nama_metode
):

    path = os.path.join(
        OUTPUT_DIR,
        f"{nama_metode.lower()}_parameter_normalisasi.json"
    )

    parameter = {
        "metode": nama_metode,
        "mean": mean_fitur,
        "std": std_fitur,
        "nama_fitur": nama_fitur
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            parameter,
            file,
            indent=4
        )

    print(
        f"Parameter normalisasi tersimpan: {path}"
    )


# ==========================================
# 8. MENYIMPAN DATA PREPROCESSING
# ==========================================

def simpan_csv(
    data,
    nama_fitur,
    nama_file
):

    path = os.path.join(
        OUTPUT_DIR,
        nama_file
    )

    header = [
        "nama_file",
        "label"
    ] + nama_fitur

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        for baris in data:

            writer.writerow(
                [baris["nama_file"], baris["label"]]
                + baris["fitur"]
            )

    print(f"File tersimpan: {path}")


# ==========================================
# 9. PROSES PREPROCESSING
# ==========================================

def proses_preprocessing(
    nama_metode,
    path_input
):

    print("\n====================================")
    print(f"PREPROCESSING {nama_metode}")
    print("====================================")

    # Membaca dataset
    data, nama_fitur = baca_csv(path_input)

    print(f"Jumlah data awal : {len(data)}")
    print(f"Jumlah fitur awal: {len(nama_fitur)}")

    # Pembagian dataset
    train, validasi, test = bagi_dataset(data)

    print("\nPembagian Dataset:")
    print(f"Train    : {len(train)}")
    print(f"Validasi : {len(validasi)}")
    print(f"Test    : {len(test)}")

    # Feature Reduction
    (
        fitur_dipertahankan,
        fitur_dihapus,
        nama_fitur_terpilih
    ) = seleksi_fitur(
        train,
        validasi,
        test,
        nama_fitur
    )

    print("\nFeature Reduction:")
    print(
        f"Fitur dipertahankan: "
        f"{len(fitur_dipertahankan)}"
    )
    print(
        f"Fitur dihapus      : "
        f"{len(fitur_dihapus)}"
    )

    # Normalisasi Z-Score
    mean_fitur, std_fitur = normalisasi_zscore(
        train,
        validasi,
        test
    )

    print("\nNormalisasi Z-Score selesai.")

    # Menyimpan parameter normalisasi
    simpan_parameter_normalisasi(
        mean_fitur,
        std_fitur,
        nama_fitur_terpilih,
        nama_metode
    )

    # Menyimpan data train
    simpan_csv(
        train,
        nama_fitur_terpilih,
        f"{nama_metode.lower()}_train.csv"
    )

    # Menyimpan data validasi
    simpan_csv(
        validasi,
        nama_fitur_terpilih,
        f"{nama_metode.lower()}_validasi.csv"
    )

    # Menyimpan data test
    simpan_csv(
        test,
        nama_fitur_terpilih,
        f"{nama_metode.lower()}_test.csv"
    )

    print(
        f"\nPreprocessing {nama_metode} selesai!"
    )


# ==========================================
# 10. MENJALANKAN MFCC DAN MEL
# ==========================================

if __name__ == "__main__":

    proses_preprocessing(
        "MFCC",
        INPUT_MFCC
    )

    proses_preprocessing(
        "Mel",
        INPUT_MEL
    )

    print("\nSEMUA PREPROCESSING SELESAI.")