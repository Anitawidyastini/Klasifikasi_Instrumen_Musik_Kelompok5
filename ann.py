
import csv
import math
import random
import os

from config import HASIL_DIR


# ==========================================
# 1. KONFIGURASI ANN
# ==========================================

JUMLAH_NEURON_HIDDEN = 32
JUMLAH_OUTPUT = 3

LEARNING_RATE = 0.01
EPOCH = 100
BATCH_SIZE = 16

LABEL_MAPPING = {
    "Acoustic_Guitar": 0,
    "Bass_Guitar": 1,
    "Banjo": 2
}

NAMA_KELAS = [
    "Acoustic_Guitar",
    "Bass_Guitar",
    "Banjo"
]


# ==========================================
# 2. MEMBACA DATA TRAINING
# ==========================================

def baca_data_training(path):

    data = []

    with open(path, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        kolom_fitur = [
            kolom for kolom in reader.fieldnames
            if kolom not in ["nama_file", "label"]
        ]

        for baris in reader:

            fitur = [
                float(baris[kolom])
                for kolom in kolom_fitur
            ]

            label = LABEL_MAPPING[baris["label"]]

            data.append({
                "fitur": fitur,
                "label": label
            })

    return data, len(kolom_fitur)


# ==========================================
# 3. ONE-HOT ENCODING LABEL
# ==========================================

def one_hot(label):

    hasil = [0.0] * JUMLAH_OUTPUT

    hasil[label] = 1.0

    return hasil


# ==========================================
# 4. INISIALISASI WEIGHT DAN BIAS
# ==========================================

def inisialisasi_parameter(jumlah_input):

    random.seed(42)

    # Weight input menuju hidden layer
    batas_w1 = math.sqrt(
        2 / (jumlah_input + JUMLAH_NEURON_HIDDEN)
    )

    W1 = [
        [
            random.uniform(-batas_w1, batas_w1)
            for _ in range(jumlah_input)
        ]
        for _ in range(JUMLAH_NEURON_HIDDEN)
    ]

    b1 = [0.0] * JUMLAH_NEURON_HIDDEN

    # Weight hidden menuju output layer
    batas_w2 = math.sqrt(
        2 / (JUMLAH_NEURON_HIDDEN + JUMLAH_OUTPUT)
    )

    W2 = [
        [
            random.uniform(-batas_w2, batas_w2)
            for _ in range(JUMLAH_NEURON_HIDDEN)
        ]
        for _ in range(JUMLAH_OUTPUT)
    ]

    b2 = [0.0] * JUMLAH_OUTPUT

    return W1, b1, W2, b2


# ==========================================
# 5. FUNGSI AKTIVASI ReLU
# ==========================================

def relu(x):

    if x > 0:
        return x

    return 0.0


# ==========================================
# 6. FUNGSI AKTIVASI SOFTMAX
# ==========================================

def softmax(z):

    nilai_maksimum = max(z)

    eksponensial = [
        math.exp(x - nilai_maksimum)
        for x in z
    ]

    total = sum(eksponensial)

    return [
        nilai / total
        for nilai in eksponensial
    ]


# ==========================================
# 7. FORWARD PROPAGATION
# ==========================================

def forward_propagation(
    fitur,
    W1,
    b1,
    W2,
    b2
):

    # Input menuju hidden layer
    z1 = []

    for i in range(JUMLAH_NEURON_HIDDEN):

        total = b1[i]

        for j in range(len(fitur)):

            total += W1[i][j] * fitur[j]

        z1.append(total)

    # Aktivasi hidden layer
    a1 = [
        relu(nilai)
        for nilai in z1
    ]

    # Hidden layer menuju output layer
    z2 = []

    for i in range(JUMLAH_OUTPUT):

        total = b2[i]

        for j in range(JUMLAH_NEURON_HIDDEN):

            total += W2[i][j] * a1[j]

        z2.append(total)

    # Probabilitas output
    a2 = softmax(z2)

    return z1, a1, z2, a2


# ==========================================
# 8. LOSS FUNCTION
# ==========================================

def categorical_cross_entropy(prediksi, target):

    epsilon = 1e-12

    loss = 0.0

    for i in range(len(target)):

        loss -= target[i] * math.log(
            max(prediksi[i], epsilon)
        )

    return loss


# ==========================================
# 9. BACKPROPAGATION DAN UPDATE PARAMETER
# ==========================================

def backpropagation(
    fitur,
    label,
    z1,
    a1,
    a2,
    W1,
    b1,
    W2,
    b2
):

    target = one_hot(label)

    # --------------------------------------
    # Gradien output layer
    # Softmax + Cross Entropy
    # --------------------------------------

    delta2 = [
        a2[i] - target[i]
        for i in range(JUMLAH_OUTPUT)
    ]

    # Gradien W2 dan b2
    grad_W2 = [
        [
            delta2[i] * a1[j]
            for j in range(JUMLAH_NEURON_HIDDEN)
        ]
        for i in range(JUMLAH_OUTPUT)
    ]

    grad_b2 = delta2[:]

    # --------------------------------------
    # Gradien hidden layer
    # --------------------------------------

    delta1 = []

    for j in range(JUMLAH_NEURON_HIDDEN):

        total = 0.0

        for i in range(JUMLAH_OUTPUT):

            total += W2[i][j] * delta2[i]

        # Turunan ReLU
        if z1[j] <= 0:
            total = 0.0

        delta1.append(total)

    # Gradien W1 dan b1
    grad_W1 = [
        [
            delta1[i] * fitur[j]
            for j in range(len(fitur))
        ]
        for i in range(JUMLAH_NEURON_HIDDEN)
    ]

    grad_b1 = delta1[:]

    # --------------------------------------
    # UPDATE WEIGHT DAN BIAS
    # --------------------------------------

    for i in range(JUMLAH_NEURON_HIDDEN):

        for j in range(len(fitur)):

            W1[i][j] -= (
                LEARNING_RATE * grad_W1[i][j]
            )

        b1[i] -= (
            LEARNING_RATE * grad_b1[i]
        )

    for i in range(JUMLAH_OUTPUT):

        for j in range(JUMLAH_NEURON_HIDDEN):

            W2[i][j] -= (
                LEARNING_RATE * grad_W2[i][j]
            )

        b2[i] -= (
            LEARNING_RATE * grad_b2[i]
        )


# ==========================================
# 10. UJI SATU KALI UPDATE WEIGHT
# ==========================================

def uji_update_weight(
    fitur,
    label,
    W1,
    b1,
    W2,
    b2
):

    # Forward propagation sebelum update
    z1, a1, z2, a2 = forward_propagation(
        fitur,
        W1,
        b1,
        W2,
        b2
    )

    target = one_hot(label)

    loss_awal = categorical_cross_entropy(
        a2,
        target
    )

    prediksi_awal = a2.index(max(a2))

    # Backpropagation
    backpropagation(
        fitur,
        label,
        z1,
        a1,
        a2,
        W1,
        b1,
        W2,
        b2
    )

    # Forward propagation setelah update
    z1_baru, a1_baru, z2_baru, a2_baru = (
        forward_propagation(
            fitur,
            W1,
            b1,
            W2,
            b2
        )
    )

    loss_baru = categorical_cross_entropy(
        a2_baru,
        target
    )

    prediksi_baru = a2_baru.index(
        max(a2_baru)
    )

    print("\n===================================")
    print("HASIL UPDATE WEIGHT")
    print("===================================")

    print("Loss sebelum update :", round(loss_awal, 6))
    print("Loss setelah update :", round(loss_baru, 6))

    print(
        "Prediksi sebelum   :",
        NAMA_KELAS[prediksi_awal]
    )

    print(
        "Prediksi setelah   :",
        NAMA_KELAS[prediksi_baru]
    )

    print(
        "Label sebenarnya   :",
        NAMA_KELAS[label]
    )


# ==========================================
# 11. PENGUJIAN ANN
# ==========================================

if __name__ == "__main__":

    path_train = os.path.join(
        HASIL_DIR,
        "preprocessing",
        "mfcc_train.csv"
    )

    data_train, jumlah_input = baca_data_training(
        path_train
    )

    print("===================================")
    print("PENGUJIAN ANN")
    print("===================================")

    print("Jumlah data training:", len(data_train))
    print("Jumlah fitur input:", jumlah_input)

    # Inisialisasi parameter
    W1, b1, W2, b2 = inisialisasi_parameter(
        jumlah_input
    )

    # Mengambil satu sampel
    sampel = data_train[0]

    fitur = sampel["fitur"]
    label = sampel["label"]

    # Forward propagation
    z1, a1, z2, a2 = forward_propagation(
        fitur,
        W1,
        b1,
        W2,
        b2
    )

    print("\nLabel sebenarnya:", NAMA_KELAS[label])

    print("\nProbabilitas sebelum update:")

    for i in range(JUMLAH_OUTPUT):

        print(
            NAMA_KELAS[i],
            ":",
            round(a2[i] * 100, 2),
            "%"
        )

    print(
        "\nPrediksi sebelum update:",
        NAMA_KELAS[a2.index(max(a2))]
    )

    # Menghitung loss
    loss = categorical_cross_entropy(
        a2,
        one_hot(label)
    )

    print("Loss awal:", round(loss, 6))

    # Update weight satu kali
    uji_update_weight(
        fitur,
        label,
        W1,
        b1,
        W2,
        b2
    )

    print("\nPengujian backpropagation berhasil!")