
import os
import json
import csv

from config import HASIL_DIR
from ann import forward_propagation


# ==========================================
# 1. KONFIGURASI EVALUASI
# ==========================================

FOLDER_MODEL = os.path.join(
    HASIL_DIR,
    "model"
)

FILE_MODEL = os.path.join(
    FOLDER_MODEL,
    "model_mel_terbaik.json"
)

FILE_TEST = os.path.join(
    HASIL_DIR,
    "preprocessing",
    "mel_test.csv"
)

FILE_HASIL = os.path.join(
    FOLDER_MODEL,
    "evaluasi_mel_test.csv"
)

NAMA_KELAS = [
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Banjo"
]

JUMLAH_KELAS = len(NAMA_KELAS)


# ==========================================
# 2. MEMUAT MODEL TERBAIK
# ==========================================

def muat_model(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        model = json.load(file)

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    print("Model berhasil dimuat.")
    print("Metode        :", model["metode"])
    print("Epoch terbaik :", model["epoch_terbaik"])
    print("Validation Loss:", model["val_loss_terbaik"])

    return W1, b1, W2, b2, model


# ==========================================
# 3. MEMBACA DATA TEST
# ==========================================

def baca_data_test(path):

    data = []

    mapping_label = {
        "Acoustic_Guitar": 0,
        "Bass_Guitar": 1,
        "Banjo": 2
    }

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for baris in reader:

            nama_file = baris["nama_file"]
            label = baris["label"]

            fitur = [
                float(baris[k])
                for k in baris
                if k not in ["nama_file", "label"]
            ]

            data.append({
                "nama_file": nama_file,
                "fitur": fitur,
                "label": mapping_label[label]
            })

    return data


# ==========================================
# 4. PERHITUNGAN METRIK EVALUASI
# ==========================================

def hitung_metrik(
    confusion_matrix,
    jumlah_data
):

    total_benar = sum(
        confusion_matrix[i][i]
        for i in range(JUMLAH_KELAS)
    )

    accuracy = total_benar / jumlah_data

    precision_per_kelas = []
    recall_per_kelas = []
    f1_per_kelas = []

    for i in range(JUMLAH_KELAS):

        true_positive = confusion_matrix[i][i]

        false_positive = sum(
            confusion_matrix[j][i]
            for j in range(JUMLAH_KELAS)
            if j != i
        )

        false_negative = sum(
            confusion_matrix[i][j]
            for j in range(JUMLAH_KELAS)
            if j != i
        )

        # Precision
        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = (
                true_positive /
                (true_positive + false_positive)
            )

        # Recall
        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = (
                true_positive /
                (true_positive + false_negative)
            )

        # F1-Score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall /
                (precision + recall)
            )

        precision_per_kelas.append(precision)
        recall_per_kelas.append(recall)
        f1_per_kelas.append(f1)

    # Macro Average
    macro_precision = sum(
        precision_per_kelas
    ) / JUMLAH_KELAS

    macro_recall = sum(
        recall_per_kelas
    ) / JUMLAH_KELAS

    macro_f1 = sum(
        f1_per_kelas
    ) / JUMLAH_KELAS

    return (
        accuracy,
        precision_per_kelas,
        recall_per_kelas,
        f1_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ==========================================
# 5. EVALUASI MODEL PADA DATA TEST
# ==========================================

def evaluasi_test(
    data_test,
    W1,
    b1,
    W2,
    b2
):

    confusion_matrix = [
        [0] * JUMLAH_KELAS
        for _ in range(JUMLAH_KELAS)
    ]

    total_loss = 0.0

    for sampel in data_test:

        _, _, _, prediksi = forward_propagation(
            sampel["fitur"],
            W1,
            b1,
            W2,
            b2
        )

        label_asli = sampel["label"]

        label_prediksi = prediksi.index(
            max(prediksi)
        )

        # Confusion Matrix
        confusion_matrix[
            label_asli
        ][
            label_prediksi
        ] += 1

        # Cross Entropy
        epsilon = 1e-12

        total_loss -= (
            __import__("math").log(
                max(prediksi[label_asli], epsilon)
            )
        )

    test_loss = total_loss / len(data_test)

    hasil_metrik = hitung_metrik(
        confusion_matrix,
        len(data_test)
    )

    return (
        confusion_matrix,
        test_loss,
        hasil_metrik
    )


# ==========================================
# 6. MENYIMPAN HASIL EVALUASI
# ==========================================

def simpan_hasil(
    confusion_matrix,
    test_loss,
    hasil_metrik
):

    (
        accuracy,
        precision_per_kelas,
        recall_per_kelas,
        f1_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1
    ) = hasil_metrik

    os.makedirs(
        FOLDER_MODEL,
        exist_ok=True
    )

    with open(
        FILE_HASIL,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Jenis",
            "Kelas",
            "Accuracy",
            "Test_Loss",
            "Precision",
            "Recall",
            "F1_Score"
        ])

        writer.writerow([
            "Overall",
            "Semua Kelas",
            accuracy,
            test_loss,
            macro_precision,
            macro_recall,
            macro_f1
        ])

        for i in range(JUMLAH_KELAS):

            writer.writerow([
                "Per Kelas",
                NAMA_KELAS[i],
                "",
                "",
                precision_per_kelas[i],
                recall_per_kelas[i],
                f1_per_kelas[i]
            ])

        writer.writerow([])
        writer.writerow(["Confusion Matrix"])
        writer.writerow(["Aktual / Prediksi"] + NAMA_KELAS)

        for i in range(JUMLAH_KELAS):

            writer.writerow([
                NAMA_KELAS[i]
            ] + confusion_matrix[i])

    print("\nHasil evaluasi berhasil disimpan.")


# ==========================================
# 7. PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    # Memuat model terbaik
    W1, b1, W2, b2, model = muat_model(
        FILE_MODEL
    )

    # Membaca data test
    data_test = baca_data_test(
        FILE_TEST
    )

    print("\n===================================")
    print("EVALUASI ANN - MEL DATA TEST")
    print("===================================")

    print("Jumlah data test:", len(data_test))
    print("Jumlah fitur    :", len(data_test[0]["fitur"]))

    # Melakukan evaluasi
    confusion_matrix, test_loss, hasil_metrik = evaluasi_test(
        data_test,
        W1,
        b1,
        W2,
        b2
    )

    (
        accuracy,
        precision_per_kelas,
        recall_per_kelas,
        f1_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1
    ) = hasil_metrik

    # ----------------------------------
    # MENAMPILKAN CONFUSION MATRIX
    # ----------------------------------

    print("\nCONFUSION MATRIX")

    print(
        f"{'Aktual / Prediksi':<22}",
        *[f"{kelas:>15}" for kelas in NAMA_KELAS]
    )

    for i in range(JUMLAH_KELAS):

        print(
            f"{NAMA_KELAS[i]:<22}",
            *[f"{nilai:>15}" for nilai in confusion_matrix[i]]
        )

    # ----------------------------------
    # MENAMPILKAN HASIL EVALUASI
    # ----------------------------------

    print("\nHASIL EVALUASI DATA TEST")

    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Test Loss      : {test_loss:.6f}")
    print(f"Macro Precision: {macro_precision * 100:.2f}%")
    print(f"Macro Recall   : {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")

    # ----------------------------------
    # HASIL PER KELAS
    # ----------------------------------

    print("\nHASIL PER KELAS")

    for i in range(JUMLAH_KELAS):

        jumlah = sum(confusion_matrix[i])

        print(f"\n{NAMA_KELAS[i]}")

        print(
            f"Precision: "
            f"{precision_per_kelas[i] * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{recall_per_kelas[i] * 100:.2f}%"
        )

        print(
            f"F1-Score : "
            f"{f1_per_kelas[i] * 100:.2f}%"
        )

        print(f"Jumlah   : {jumlah}")

    # ----------------------------------
    # SIMPAN HASIL
    # ----------------------------------

    simpan_hasil(
        confusion_matrix,
        test_loss,
        hasil_metrik
    )

    print("\nEvaluasi Mel-Spectrogram selesai.")