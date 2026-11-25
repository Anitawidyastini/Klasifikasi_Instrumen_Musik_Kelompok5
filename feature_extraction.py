# ==========================================
# EKSTRAKSI FITUR AUDIO
# ==========================================

from audio_processing import (
    baca_audio,
    buat_frame,
    terapkan_window,
    buat_mel_filter_bank,
    ekstrasi_mfcc_dan_mel_seluruh_frame,
    hitung_mean_std_mfcc,
    hitung_mean_std_mel_spectrogram
)

from config import (
    FRAME_LENGTH,
    HOP_LENGTH,
    JUMLAH_FILTER_MEL
)


def ekstraksi_fitur_audio(path_audio):

    # Membaca audio
    sampel_audio, sample_rate, jumlah_channel, bit_depth, durasi = baca_audio(
        path_audio
    )

    # Membuat frame
    frames = buat_frame(
        sampel_audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    # Windowing
    frames_window = terapkan_window(frames)

    # Membuat Mel Filter Bank
    filter_bank, titik_bin = buat_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=FRAME_LENGTH,
        jumlah_filter=JUMLAH_FILTER_MEL
    )

    # ==========================================
    # EKSTRAKSI MFCC DAN MEL BERSAMA
    # ==========================================

    seluruh_mfcc, seluruh_mel_spectrogram = (
        ekstrasi_mfcc_dan_mel_seluruh_frame(
            frames_window,
            filter_bank
        )
    )

    # ==========================================
    # MEAN DAN STANDAR DEVIASI MFCC
    # ==========================================

    mean_mfcc, std_mfcc, fitur_mfcc = hitung_mean_std_mfcc(
        seluruh_mfcc
    )

    # ==========================================
    # MEAN DAN STANDAR DEVIASI MEL-SPECTROGRAM
    # ==========================================

    mean_mel, std_mel, fitur_mel_spectrogram = (
        hitung_mean_std_mel_spectrogram(
            seluruh_mel_spectrogram
        )
    )

    return fitur_mfcc, fitur_mel_spectrogram