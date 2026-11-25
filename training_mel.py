
import os
import random
import json
import csv
import copy

from config import HASIL_DIR

from ann import (
    baca_data_training,
    inisialisasi_parameter,
    forward_propagation,
    one_hot,
    categorical_cross_entropy,
    JUMLAH_NEURON_HIDDEN,
    JUMLAH_OUTPUT
)


# ==========================================
# 1. KONFIGURASI TRAINING
# ==========================================

LEARNING_RATE = 0.01
EPOCH = 20
BATCH_SIZE = 16
RANDOM_SEED = 42

# Konfigurasi penyimpanan model
FOLDER_MODEL = os.path.join(HASIL_DIR, "model")

FILE_MODEL = os.path.join(
    FOLDER_MODEL,
    "model_mel_terbaik.json"
)

FILE_HISTORY = os.path.join(
    FOLDER_MODEL,
    "history_mel.csv"
)


# ==========================================
# 2. MENGHITUNG GRADIENT SATU DATA
# ==========================================

def hitung_gradien(
    fitur,
    label,
    W1,
    b1,
    W2,
    b2
):

    z1, a1, z2, a2 = forward_propagation(
        fitur,
        W1,
        b1,
        W2,
        b2
    )

    target = one_hot(label)

    # Gradien output layer
    delta2 = [
        a2[i] - target[i]
        for i in range(JUMLAH_OUTPUT)
    ]

    grad_W2 = [
        [
            delta2[i] * a1[j]
            for j in range(JUMLAH_NEURON_HIDDEN)
        ]
        for i in range(JUMLAH_OUTPUT)
    ]

    grad_b2 = delta2[:]

    # Gradien hidden layer
    delta1 = []

    for j in range(JUMLAH_NEURON_HIDDEN):

        total = 0.0

        for i in range(JUMLAH_OUTPUT):
            total += W2[i][j] * delta2[i]

        # Turunan fungsi aktivasi ReLU
        if z1[j] <= 0:
            total = 0.0

        delta1.append(total)

    grad_W1 = [
        [
            delta1[i] * fitur[j]
            for j in range(len(fitur))
        ]
        for i in range(JUMLAH_NEURON_HIDDEN)
    ]

    grad_b1 = delta1[:]

    return grad_W1, grad_b1, grad_W2, grad_b2


# ==========================================
# 3. UPDATE PARAMETER DENGAN GRADIENT BATCH
# ==========================================

def update_batch(
    batch,
    W1,
    b1,
    W2,
    b2
):

    jumlah_input = len(batch[0]["fitur"])

    # Inisialisasi akumulasi gradien W1 dan b1
    total_grad_W1 = [
        [0.0] * jumlah_input
        for _ in range(JUMLAH_NEURON_HIDDEN)
    ]

    total_grad_b1 = [
        0.0
        for _ in range(JUMLAH_NEURON_HIDDEN)
    ]

    # Inisialisasi akumulasi gradien W2 dan b2
    total_grad_W2 = [
        [0.0] * JUMLAH_NEURON_HIDDEN
        for _ in range(JUMLAH_OUTPUT)
    ]

    total_grad_b2 = [
        0.0
        for _ in range(JUMLAH_OUTPUT)
    ]

    # Menghitung gradien setiap data dalam batch
    for sampel in batch:

        grad_W1, grad_b1, grad_W2, grad_b2 = hitung_gradien(
            sampel["fitur"],
            sampel["label"],
            W1,
            b1,
            W2,
            b2
        )

        # Akumulasi gradien W1 dan b1
        for i in range(JUMLAH_NEURON_HIDDEN):

            for j in range(jumlah_input):
                total_grad_W1[i][j] += grad_W1[i][j]

            total_grad_b1[i] += grad_b1[i]

        # Akumulasi gradien W2 dan b2
        for i in range(JUMLAH_OUTPUT):

            for j in range(JUMLAH_NEURON_HIDDEN):
                total_grad_W2[i][j] += grad_W2[i][j]

            total_grad_b2[i] += grad_b2[i]

    # Menghitung rata-rata gradien batch
    ukuran_batch = len(batch)

    # Update W1 dan b1
    for i in range(JUMLAH_NEURON_HIDDEN):

        for j in range(jumlah_input):

            rata_grad = total_grad_W1[i][j] / ukuran_batch

            W1[i][j] -= LEARNING_RATE * rata_grad

        rata_grad_bias = total_grad_b1[i] / ukuran_batch

        b1[i] -= LEARNING_RATE * rata_grad_bias

    # Update W2 dan b2
    for i in range(JUMLAH_OUTPUT):

        for j in range(JUMLAH_NEURON_HIDDEN):

            rata_grad = total_grad_W2[i][j] / ukuran_batch

            W2[i][j] -= LEARNING_RATE * rata_grad

        rata_grad_bias = total_grad_b2[i] / ukuran_batch

        b2[i] -= LEARNING_RATE * rata_grad_bias

    return W1, b1, W2, b2


# ==========================================
# 4. EVALUASI DATA
# ==========================================

def evaluasi(data, W1, b1, W2, b2):

    if len(data) == 0:
        raise ValueError("Data evaluasi kosong.")

    total_loss = 0.0
    benar = 0

    for sampel in data:

        _, _, _, prediksi = forward_propagation(
            sampel["fitur"],
            W1,
            b1,
            W2,
            b2
        )

        target = one_hot(sampel["label"])

        total_loss += categorical_cross_entropy(
            prediksi,
            target
        )

        kelas_prediksi = prediksi.index(max(prediksi))

        if kelas_prediksi == sampel["label"]:
            benar += 1

    loss_rata = total_loss / len(data)
    akurasi = benar / len(data) * 100

    return loss_rata, akurasi


# ==========================================
# 5. MENYIMPAN MODEL TERBAIK
# ==========================================

def simpan_model(
    W1,
    b1,
    W2,
    b2,
    epoch_terbaik,
    val_loss_terbaik
):

    os.makedirs(FOLDER_MODEL, exist_ok=True)

    model = {
        "metode": "Mel-Spectrogram",

        "arsitektur": {
            "input": len(W1[0]),
            "hidden": JUMLAH_NEURON_HIDDEN,
            "output": JUMLAH_OUTPUT
        },

        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epoch_maksimum": EPOCH,
        "epoch_terbaik": epoch_terbaik,
        "val_loss_terbaik": val_loss_terbaik,

        "label_mapping": {
            "Acoustic_Guitar": 0,
            "Bass_Guitar": 1,
            "Banjo": 2
        },

        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    with open(
        FILE_MODEL,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(model, file)

    print(
        f"Model terbaik tersimpan "
        f"(epoch {epoch_terbaik})."
    )


# ==========================================
# 6. MENYIMPAN RIWAYAT TRAINING
# ==========================================

def simpan_history(history):

    os.makedirs(FOLDER_MODEL, exist_ok=True)

    with open(
        FILE_HISTORY,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])

        writer.writerows(history)

    print("History training berhasil disimpan.")


# ==========================================
# 7. PROSES TRAINING ANN
# ==========================================

def training_ann(
    data_train,
    data_validasi,
    W1,
    b1,
    W2,
    b2
):

    random.seed(RANDOM_SEED)

    history = []

    val_loss_terbaik = float("inf")
    epoch_terbaik = 0

    W1_terbaik = None
    b1_terbaik = None
    W2_terbaik = None
    b2_terbaik = None

    for epoch in range(1, EPOCH + 1):

        # Pengacakan urutan data training
        random.shuffle(data_train)

        # Training menggunakan mini-batch
        for i in range(0, len(data_train), BATCH_SIZE):

            batch = data_train[i:i + BATCH_SIZE]

            W1, b1, W2, b2 = update_batch(
                batch,
                W1,
                b1,
                W2,
                b2
            )

        # Evaluasi data training
        train_loss, train_accuracy = evaluasi(
            data_train,
            W1,
            b1,
            W2,
            b2
        )

        # Evaluasi data validasi
        val_loss, val_accuracy = evaluasi(
            data_validasi,
            W1,
            b1,
            W2,
            b2
        )

        # Menampilkan hasil setiap epoch
        print(
            f"Epoch {epoch}/{EPOCH} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Menyimpan history training
        history.append([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy
        ])

        # Menentukan model terbaik berdasarkan validation loss
        if val_loss < val_loss_terbaik:

            val_loss_terbaik = val_loss
            epoch_terbaik = epoch

            W1_terbaik = copy.deepcopy(W1)
            b1_terbaik = copy.deepcopy(b1)
            W2_terbaik = copy.deepcopy(W2)
            b2_terbaik = copy.deepcopy(b2)

            # Menyimpan model apabila lebih baik
            simpan_model(
                W1_terbaik,
                b1_terbaik,
                W2_terbaik,
                b2_terbaik,
                epoch_terbaik,
                val_loss_terbaik
            )

    # Menyimpan riwayat training
    simpan_history(history)

    # Menampilkan hasil akhir
    print("\n===================================")
    print("TRAINING SELESAI")
    print("===================================")

    print(f"Epoch terbaik         : {epoch_terbaik}")
    print(f"Validation Loss terbaik: {val_loss_terbaik:.6f}")

    return (
        W1_terbaik,
        b1_terbaik,
        W2_terbaik,
        b2_terbaik
    )


# ==========================================
# 8. PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    folder_preprocessing = os.path.join(
        HASIL_DIR,
        "preprocessing"
    )

    path_train = os.path.join(
        folder_preprocessing,
        "mel_train.csv"
    )

    path_validasi = os.path.join(
        folder_preprocessing,
        "mel_validasi.csv"
    )

    # Membaca data training
    data_train, jumlah_input = baca_data_training(
        path_train
    )

    # Membaca data validasi
    data_validasi, _ = baca_data_training(
        path_validasi
    )

    print("===================================")
    print("TRAINING ANN - MEL-SPECTROGRAM")
    print("===================================")

    print("Data training :", len(data_train))
    print("Data validasi:", len(data_validasi))
    print("Jumlah input :", jumlah_input)
    print("Batch size  :", BATCH_SIZE)
    print("Epoch       :", EPOCH)
    print("Learning rate:", LEARNING_RATE)

    # Inisialisasi parameter ANN
    W1, b1, W2, b2 = inisialisasi_parameter(
        jumlah_input
    )

    # Memulai training
    W1, b1, W2, b2 = training_ann(
        data_train,
        data_validasi,
        W1,
        b1,
        W2,
        b2
    )

    print("\nTraining Mel-Spectrogram selesai.")