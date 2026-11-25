
import os
import csv

from config import HASIL_DIR

def cek_csv(nama_file):
    path = os.path.join(HASIL_DIR, nama_file)

    print(f"\n{'=' * 50}")
    print(f"PEMERIKSAAN: {nama_file}")
    print(f"{'=' * 50}")

    if not os.path.exists(path):
        print("File tidak ditemukan!")
        return

    jumlah_data = 0
    jumlah_kolom = 0
    data_kosong = 0
    distribusi = {}

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        header = next(reader, None)

        if header is None:
            print("File CSV kosong!")
            return

        jumlah_kolom = len(header)

        for baris in reader:
            if not baris:
                continue

            jumlah_data += 1

            if any(nilai.strip() == "" for nilai in baris):
                data_kosong += 1

            label = baris[-1]
            distribusi[label] = distribusi.get(label, 0) + 1

    print(f"Jumlah data       : {jumlah_data}")
    print(f"Jumlah kolom     : {jumlah_kolom}")
    print(f"Data kosong      : {data_kosong}")
    print(f"Distribusi kelas : {distribusi}")
    print(f"Header           : {header}")


if __name__ == "__main__":
    cek_csv("fitur_mfcc_dataset.csv")
    cek_csv("fitur_mel_dataset.csv")