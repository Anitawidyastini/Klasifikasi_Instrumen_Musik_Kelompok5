# Klasifikasi Instrumen Musik dengan ANN

Project ini dibuat untuk mengklasifikasikan instrumen musik dari file audio menggunakan Artificial Neural Network (ANN). Klasifikasi dilakukan dengan memanfaatkan fitur audio yang diekstraksi menggunakan MFCC dan Mel-Spectrogram, kemudian hasilnya digunakan sebagai input untuk model ANN.

Project ini membandingkan kedua metode ekstraksi fitur tersebut dalam mengenali tiga jenis instrumen musik, yaitu Acoustic Guitar, Bass Guitar, dan Banjo. Aplikasi dibuat menggunakan Streamlit agar proses klasifikasi dapat dilakukan melalui tampilan sederhana tanpa harus menjalankan program secara manual.

## Instrumen yang Diklasifikasikan

- Acoustic Guitar
- Bass Guitar
- Banjo

## Metode yang Digunakan

**MFCC (Mel-Frequency Cepstral Coefficients)**

Digunakan untuk mengambil karakteristik audio berdasarkan representasi frekuensi pada skala Mel. Fitur yang dihasilkan kemudian dirangkum menggunakan mean dan standard deviation.

**Mel-Spectrogram**

Digunakan untuk merepresentasikan energi audio pada frekuensi tertentu berdasarkan skala Mel. Hasil ekstraksi dirangkum menjadi fitur yang digunakan sebagai input klasifikasi.

**ANN (Artificial Neural Network)**

Digunakan untuk mempelajari pola dari fitur audio dan menentukan jenis instrumen musik berdasarkan data yang telah dilatih sebelumnya.

## Menjalankan Project

### 1. Clone Repository

Buka terminal, kemudian jalankan perintah berikut untuk mengambil project dari GitHub.

```bash
git clone https://github.com/Anitawidyastini/Klasifikasi_Instrumen_Musik_Kelompok5.git
```

### 2. Masuk ke Folder Project

```bash
cd Klasifikasi_Instrumen_Musik_Kelompok5
```

### 3. Membuat Environment

Project ini menggunakan Python 3.11. Jika menggunakan Conda, buat environment baru dengan perintah:

```bash
conda create -n musik_ann python=3.11
```

Setelah selesai, aktifkan environment:

```bash
conda activate musik_ann
```

### 4. Install Library

Install library yang dibutuhkan oleh project:

```bash
pip install -r requirements.txt
```

Tunggu sampai proses instalasi selesai.

### 5. Jalankan Aplikasi

Untuk menjalankan aplikasi klasifikasi, gunakan perintah:

```bash
streamlit run app.py
```

Setelah dijalankan, aplikasi akan terbuka melalui browser. Jika tidak terbuka secara otomatis, akses:

http://localhost:8501

## Cara Menggunakan Aplikasi

1. Buka aplikasi melalui browser.
2. Pilih metode ekstraksi fitur yang ingin digunakan, yaitu MFCC atau Mel-Spectrogram.
3. Upload file audio yang ingin diuji.
4. Tunggu proses ekstraksi fitur dan klasifikasi selesai.
5. Hasil prediksi instrumen musik akan ditampilkan pada aplikasi.

## Struktur Folder

```text
Klasifikasi_Instrumen_Musik_Kelompok5/
│
├── app.py
├── ann.py
├── audio_processing.py
├── feature_extraction.py
├── preprocessing.py
├── config.py
├── requirements.txt
│
├── training.py
├── training_mel.py
├── evaluasi_test.py
├── evaluasi_mel_test.py
│
├── hasil/
│   ├── model/
│   │   ├── model_mfcc_terbaik.json
│   │   └── model_mel_terbaik.json
│   │
│   └── preprocessing/
│       ├── mfcc_parameter_normalisasi.json
│       └── mel_parameter_normalisasi.json
│
└── dataset/
    ├── Acoustic_Guitar/
    ├── Bass_Guitar/
    └── Banjo/
```

## Catatan

- Model yang digunakan sudah disediakan di folder `hasil/model`, sehingga aplikasi dapat langsung dijalankan tanpa training ulang.
- Dataset tidak disertakan dalam repository.
- Folder dataset diperlukan apabila ingin melakukan training ulang atau pengujian menggunakan dataset.
- Pastikan Python dan library yang dibutuhkan sudah terinstall sebelum menjalankan aplikasi.
- Aplikasi dijalankan secara lokal menggunakan Streamlit.

## Kelompok 5

Pengantar Komputasi Lunak

Program Studi Informatika