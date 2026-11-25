
# ==========================================
# MEMBACA DAFTAR FILE DATASET
# ==========================================

import os

from config import DATASET_DIR, KELAS


def ambil_daftar_audio():

    daftar_audio = []

    for kelas in KELAS:

        folder_kelas = os.path.join(DATASET_DIR, kelas)

        daftar_file = os.listdir(folder_kelas)

        for nama_file in daftar_file:

            if nama_file.lower().endswith(".wav"):

                path_audio = os.path.join(folder_kelas, nama_file)

                daftar_audio.append((path_audio, kelas))

    return daftar_audio