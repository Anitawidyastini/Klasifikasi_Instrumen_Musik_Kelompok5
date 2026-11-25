
import os
import hashlib

from config import DATASET_DIR


# ==========================================
# 1. MENGHITUNG HASH FILE
# ==========================================

def hitung_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            bagian = file.read(65536)

            if not bagian:
                break

            sha256.update(bagian)

    return sha256.hexdigest()


# ==========================================
# 2. MEMERIKSA FILE AUDIO
# ==========================================

def cek_duplikasi():

    daftar_hash = {}
    jumlah_file = 0

    ekstensi_audio = (".wav",)

    print("===================================")
    print("PEMERIKSAAN DUPLIKASI AUDIO")
    print("===================================")

    for folder, _, nama_file in os.walk(DATASET_DIR):

        for nama in nama_file:

            if not nama.lower().endswith(ekstensi_audio):
                continue

            path = os.path.join(folder, nama)

            hash_file = hitung_hash(path)

            jumlah_file += 1

            if hash_file not in daftar_hash:
                daftar_hash[hash_file] = []

            daftar_hash[hash_file].append(path)

    # ======================================
    # 3. MENGELOMPOKKAN FILE DUPLIKAT
    # ======================================

    duplikat = {
        hash_file: daftar_path
        for hash_file, daftar_path in daftar_hash.items()
        if len(daftar_path) > 1
    }

    # ======================================
    # 4. MENAMPILKAN HASIL
    # ======================================

    jumlah_kelompok_duplikat = len(duplikat)

    jumlah_file_duplikat = sum(
        len(daftar_path)
        for daftar_path in duplikat.values()
    )

    jumlah_file_berlebih = sum(
        len(daftar_path) - 1
        for daftar_path in duplikat.values()
    )

    print(f"Jumlah file diperiksa    : {jumlah_file}")
    print(f"Jumlah file unik        : {len(daftar_hash)}")
    print(f"Kelompok duplikat      : {jumlah_kelompok_duplikat}")
    print(f"Total file duplikat    : {jumlah_file_duplikat}")
    print(f"File duplikat berlebih : {jumlah_file_berlebih}")

    if jumlah_kelompok_duplikat == 0:

        print("\nTidak ditemukan file audio identik.")

    else:

        print("\nDAFTAR FILE DUPLIKAT")

        for nomor, (hash_file, daftar_path) in enumerate(
            duplikat.items(),
            start=1
        ):

            print(f"\nKelompok {nomor}")
            print(f"Hash: {hash_file}")

            for path in daftar_path:

                print(f"  - {path}")

    print("\nPemeriksaan selesai.")


# ==========================================
# 5. PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    cek_duplikasi()