
import os
import json
import csv

from config import HASIL_DIR

from ann import (
    baca_data_training,
    forward_propagation,
    categorical_cross_entropy,
    one_hot,
    JUMLAH_OUTPUT
)


# ==========================================
# 1. KONFIGURASI
# ==========================================

FOLDER_MODEL = os.path.join(
    HASIL_DIR,
    "model"
)

FILE_MODEL = os.path.join(
    FOLDER_MODEL,
    "model_mfcc_terbaik.json"
)

FILE_TEST = os.path.join(
    HASIL_DIR,
    "preprocessing",
    "mfcc_test.csv"
)

FILE_HASIL = os.path.join(
    FOLDER_MODEL,
    "evaluasi_mfcc_test.csv"
)

NAMA_KELAS = [
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Banjo"
]


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
    print("Metode         :", model["metode"])
    print("Epoch terbaik  :", model["epoch_terbaik"])
    print("Validation Loss :", model["val_loss_terbaik"])

    return W1, b1, W2, b2


# ==========================================
# 3. EVALUASI DATA TEST
# ==========================================

def evaluasi_test(
    data_test,
    W1,
    b1,
    W2,
    b2
):

    jumlah_kelas = JUMLAH_OUTPUT

    # Confusion matrix
    confusion_matrix = [
        [0 for _ in range(jumlah_kelas)]
        for _ in range(jumlah_kelas)
    ]

    total_loss = 0.0
    total_benar = 0

    # --------------------------------------
    # PREDIKSI SELURUH DATA TEST
    # --------------------------------------

    for sampel in data_test:

        _, _, _, probabilitas = forward_propagation(
            sampel["fitur"],
            W1,
            b1,
            W2,
            b2
        )

        label_asli = sampel["label"]

        label_prediksi = probabilitas.index(
            max(probabilitas)
        )

        # Menghitung loss
        target = one_hot(label_asli)

        total_loss += categorical_cross_entropy(
            probabilitas,
            target
        )

        # Mengisi confusion matrix
        confusion_matrix[label_asli][label_prediksi] += 1

        # Menghitung prediksi benar
        if label_prediksi == label_asli:
            total_benar += 1

    # --------------------------------------
    # ACCURACY DAN TEST LOSS
    # --------------------------------------

    jumlah_data = len(data_test)

    accuracy = (
        total_benar / jumlah_data
    ) * 100

    test_loss = total_loss / jumlah_data

    # --------------------------------------
    # PRECISION, RECALL, DAN F1 PER KELAS
    # --------------------------------------

    hasil_per_kelas = []

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for kelas in range(jumlah_kelas):

        true_positive = confusion_matrix[kelas][kelas]

        false_positive = sum(
            confusion_matrix[i][kelas]
            for i in range(jumlah_kelas)
            if i != kelas
        )

        false_negative = sum(
            confusion_matrix[kelas][i]
            for i in range(jumlah_kelas)
            if i != kelas
        )

        # Precision
        if true_positive + false_positive > 0:

            precision = (
                true_positive /
                (true_positive + false_positive)
            ) * 100

        else:
            precision = 0.0

        # Recall
        if true_positive + false_negative > 0:

            recall = (
                true_positive /
                (true_positive + false_negative)
            ) * 100

        else:
            recall = 0.0

        # F1-Score
        if precision + recall > 0:

            f1 = (
                2 * precision * recall /
                (precision + recall)
            )

        else:
            f1 = 0.0

        hasil_per_kelas.append({
            "kelas": NAMA_KELAS[kelas],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "jumlah_data": sum(confusion_matrix[kelas])
        })

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # --------------------------------------
    # MACRO AVERAGE
    # --------------------------------------

    macro_precision = total_precision / jumlah_kelas
    macro_recall = total_recall / jumlah_kelas
    macro_f1 = total_f1 / jumlah_kelas

    return (
        accuracy,
        test_loss,
        confusion_matrix,
        hasil_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ==========================================
# 4. MENAMPILKAN CONFUSION MATRIX
# ==========================================

def tampilkan_confusion_matrix(matrix):

    print("\nCONFUSION MATRIX")
    print("-" * 75)

    print(
        f"{'Aktual / Prediksi':<22}",
        end=""
    )

    for nama in NAMA_KELAS:

        print(
            f"{nama[:12]:>15}",
            end=""
        )

    print()

    for i in range(len(NAMA_KELAS)):

        print(
            f"{NAMA_KELAS[i]:<22}",
            end=""
        )

        for j in range(len(NAMA_KELAS)):

            print(
                f"{matrix[i][j]:>15}",
                end=""
            )

        print()


# ==========================================
# 5. MENYIMPAN HASIL EVALUASI
# ==========================================

def simpan_hasil(
    hasil_per_kelas,
    macro_precision,
    macro_recall,
    macro_f1,
    accuracy,
    test_loss
):

    os.makedirs(FOLDER_MODEL, exist_ok=True)

    with open(
        FILE_HASIL,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "kelas",
            "jumlah_data",
            "precision",
            "recall",
            "f1_score"
        ])

        for hasil in hasil_per_kelas:

            writer.writerow([
                hasil["kelas"],
                hasil["jumlah_data"],
                hasil["precision"],
                hasil["recall"],
                hasil["f1_score"]
            ])

        writer.writerow([])
        writer.writerow([
            "Macro Precision",
            macro_precision
        ])

        writer.writerow([
            "Macro Recall",
            macro_recall
        ])

        writer.writerow([
            "Macro F1-Score",
            macro_f1
        ])

        writer.writerow([
            "Accuracy",
            accuracy
        ])

        writer.writerow([
            "Test Loss",
            test_loss
        ])

    print("\nHasil evaluasi berhasil disimpan.")


# ==========================================
# 6. PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    print("===================================")
    print("EVALUASI ANN - MFCC DATA TEST")
    print("===================================")

    # Memuat model terbaik
    W1, b1, W2, b2 = muat_model(
        FILE_MODEL
    )

    # Membaca data test
    data_test, jumlah_input = baca_data_training(
        FILE_TEST
    )

    print("\nJumlah data test :", len(data_test))
    print("Jumlah fitur     :", jumlah_input)

    # Evaluasi
    (
        accuracy,
        test_loss,
        confusion_matrix,
        hasil_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1
    ) = evaluasi_test(
        data_test,
        W1,
        b1,
        W2,
        b2
    )

    # --------------------------------------
    # HASIL EVALUASI
    # --------------------------------------

    tampilkan_confusion_matrix(
        confusion_matrix
    )

    print("\n===================================")
    print("HASIL EVALUASI DATA TEST")
    print("===================================")

    print(f"Accuracy       : {accuracy:.2f}%")
    print(f"Test Loss     : {test_loss:.6f}")
    print(f"Macro Precision: {macro_precision:.2f}%")
    print(f"Macro Recall  : {macro_recall:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.2f}%")

    print("\nHASIL PER KELAS")
    print("-----------------------------------")

    for hasil in hasil_per_kelas:

        print(f"\n{hasil['kelas']}")

        print(
            f"Precision: {hasil['precision']:.2f}%"
        )

        print(
            f"Recall   : {hasil['recall']:.2f}%"
        )

        print(
            f"F1-Score : {hasil['f1_score']:.2f}%"
        )

        print(
            f"Jumlah   : {hasil['jumlah_data']}"
        )

    # Simpan hasil
    simpan_hasil(
        hasil_per_kelas,
        macro_precision,
        macro_recall,
        macro_f1,
        accuracy,
        test_loss
    )

    print("\nEvaluasi MFCC selesai.")