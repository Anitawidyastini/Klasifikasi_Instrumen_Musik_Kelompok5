
# ==========================================
# PENYIMPANAN HASIL EKSTRAKSI FITUR
# ==========================================

import csv
import os


def simpan_fitur_csv(path_csv, data_fitur, nama_fitur):

    if len(data_fitur) == 0:
        return

    jumlah_fitur = len(data_fitur[0]) - 2

    header = ["nama_file", "label"]

    for i in range(jumlah_fitur):
        header.append(f"{nama_fitur}_{i + 1}")

    os.makedirs(os.path.dirname(path_csv), exist_ok=True)

    with open(path_csv, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(header)

        writer.writerows(data_fitur)