import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data
from skimage.util import random_noise

# =============================================================================
# 1. FUNGSI UTILITAS & SIMULASI NOISE
# =============================================================================

def buat_dataset_sintetis(img_bersih):
    """
    Membuat dataset sintetis dengan noise campuran: Periodik + Gaussian.
    Ini adalah citra g(x, y) yang rusak.
    """
    # Normalisasi ke [0, 1] untuk pemrosesan float
    img = img_bersih.astype(np.float32) / 255.0
    rows, cols = img.shape
    
    # --- Tambah Noise Periodik (Sinusoidal) ---
    # Kita buat noise sinusoidal di frekuensi (u=60, v=0)
    # Ini akan muncul sebagai garis-garis vertikal
    F_U = 60 # Frekuensi horizontal noise
    F_V = 0  # Frekuensi vertikal noise
    
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Buat gelombang sinus
    noise_sin = 0.2 * np.sin(2 * np.pi * (F_U * xx / cols + F_V * yy / rows))
    
    img_dengan_periodik = img + noise_sin
    
    # --- Tambah Noise Acak (Gaussian) ---
    # Tambahkan noise Gaussian di atas noise periodik
    # Ini adalah noise spasial yang harus dibersihkan CNN
    img_rusak_total = random_noise(img_dengan_periodik, mode='gaussian', var=0.01, clip=True)
    
    # Clip hasil akhir untuk memastikan rentang [0, 1]
    img_rusak_total = np.clip(img_rusak_total, 0, 1)
    
    return img_rusak_total, (F_U, F_V)

# =============================================================================
# 2. TAHAP 1: MODUL PENAPISAN FFT (RANAH FREKUENSI)
# =============================================================================

def filter_fft_notch(img_noisy, F_U, F_V, radius=5):
    """
    Menerapkan filter notch di ranah frekuensi untuk menghilangkan
    noise periodik pada frekuensi (F_U, F_V) dan (-F_U, -F_V).
    """
    
    # 1. Transformasi ke Ranah Frekuensi
    dft = np.fft.fft2(img_noisy)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img_noisy.shape
    cy, cx = rows // 2, cols // 2 # Titik tengah spektrum
    
    # 2. Buat Mask (Penapis Takik)
    # Buat mask yang semuanya '1' (meloloskan)
    mask = np.ones((rows, cols), dtype=np.uint8)
    
    # Tentukan lokasi noise di spektrum
    # Noise di (u, v) akan muncul di (cx+u, cy+v) dan (cx-u, cy-v)
    # Karena v=0, kita hanya berurusan dengan sumbu x (u)
    
    # Buat lubang '0' (memblokir) di lokasi noise
    cv2.circle(mask, (cx + F_U, cy + F_V), radius, 0, -1) # Titik noise positif
    cv2.circle(mask, (cx - F_U, cy - F_V), radius, 0, -1) # Titik noise negatif (simetris)

    # 3. Terapkan Filter
    dft_shift_filtered = dft_shift * mask
    
    # 4. Transformasi Balik ke Ranah Spasial
    idft_shift = np.fft.ifftshift(dft_shift_filtered)
    img_filtered = np.fft.ifft2(idft_shift)
    img_filtered = np.real(img_filtered) # Ambil bagian real
    
    # Clip hasil akhir
    img_filtered = np.clip(img_filtered, 0, 1)
    
    return img_filtered

# =============================================================================
# 3. TAHAP 2: SIMULASI MODUL CNN (RANAH SPASIAL)
# =============================================================================

def denoise_spasial_simulasi(img_noisy):
    """
    Mensimulasikan CNN Denoising menggunakan Nl-Means Denoising.
    Ini adalah denoiser spasial yang kuat, hebat pada noise Gaussian,
    tapi buruk pada noise periodik (sesuai hipotesis).
    """
    # OpenCV Nl-Means bekerja pada 0-255 (uint8)
    img_uint8 = (img_noisy * 255).astype(np.uint8)
    
    # 'h' adalah parameter kekuatan denoising.
    # Semakin tinggi, semakin kuat, tapi bisa blur.
    # h=10 adalah nilai yang baik untuk simulasi ini.
    hasil_denoised = cv2.fastNlMeansDenoising(img_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Kembalikan ke float [0, 1]
    return (hasil_denoised.astype(np.float32) / 255.0)

# =============================================================================
# 4. ALUR EKSEKUSI UTAMA (MAIN)
# =============================================================================

def main():
    print("Memulai simulasi restorasi citra hibrida...")
    
    # --- Persiapan Data ---
    img_bersih = data.camera() # Gunakan citra 'cameraman' standar
    img_bersih_float = img_bersih.astype(np.float32) / 255.0

    # Buat citra rusak dengan noise campuran
    img_rusak_total, (F_U, F_V) = buat_dataset_sintetis(img_bersih)
    
    print(f"Noise periodik ditambahkan pada frekuensi (u={F_U}, v={F_V})")

    # --- Jalankan Metode Pembanding & Usulan ---
    
    # 1. Baseline: FFT-Saja
    print("Menjalankan Baseline 1: FFT-Saja...")
    hasil_fft_saja = filter_fft_notch(img_rusak_total, F_U, F_V, radius=10)
    
    # 2. Baseline: "CNN-Saja" (Simulasi)
    print("Menjalankan Baseline 2: 'CNN-Saja' (Simulasi)...")
    hasil_cnn_saja = denoise_spasial_simulasi(img_rusak_total)
    
    # 3. Model Hibrida (Usulan)
    print("Menjalankan Model Hibrida (Usulan)...")
    # Tahap 1: Bersihkan noise periodik
    hasil_tahap_1_hibrida = filter_fft_notch(img_rusak_total, F_U, F_V, radius=10)
    # Tahap 2: Umpankan ke "CNN" untuk bersihkan noise acak
    hasil_final_hibrida = denoise_spasial_simulasi(hasil_tahap_1_hibrida)
    
    print("Simulasi selesai. Menghitung metrik...")

    # --- Evaluasi Kuantitatif (PSNR & SSIM) ---
    # data_range=1 karena citra kita dinormalisasi [0, 1]
    
    # 1. Rusak
    psnr_rusak = psnr(img_bersih_float, img_rusak_total, data_range=1)
    ssim_rusak = ssim(img_bersih_float, img_rusak_total, data_range=1)

    # 2. FFT-Saja
    psnr_fft = psnr(img_bersih_float, hasil_fft_saja, data_range=1)
    ssim_fft = ssim(img_bersih_float, hasil_fft_saja, data_range=1)
    
    # 3. CNN-Saja
    psnr_cnn = psnr(img_bersih_float, hasil_cnn_saja, data_range=1)
    ssim_cnn = ssim(img_bersih_float, hasil_cnn_saja, data_range=1)

    # 4. Hibrida
    psnr_hibrida = psnr(img_bersih_float, hasil_final_hibrida, data_range=1)
    ssim_hibrida = ssim(img_bersih_float, hasil_final_hibrida, data_range=1)
    
    # --- Tampilkan Hasil ---
    
    print("\n" + "="*30)
    print("HASIL EVALUASI KUANTITATIF")
    print("="*30)
    print(f"Input Rusak:\t PSNR = {psnr_rusak:.2f} dB \t| SSIM = {ssim_rusak:.4f}")
    print(f"FFT-Saja:\t PSNR = {psnr_fft:.2f} dB \t| SSIM = {ssim_fft:.4f}")
    print(f"CNN-Saja (Sim):\t PSNR = {psnr_cnn:.2f} dB \t| SSIM = {ssim_cnn:.4f}")
    print(f"HIBRIDA (Usulan):\t PSNR = {psnr_hibrida:.2f} dB \t| SSIM = {ssim_hibrida:.4f}")
    
    # --- Visualisasi ---
    plt.figure(figsize=(20, 12))
    plt.suptitle("Perbandingan Metode Restorasi Citra Hibrida", fontsize=20)
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_bersih, cmap='gray')
    plt.title(f"(a) Citra Asli (Ground Truth)")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_rusak_total, cmap='gray')
    plt.title(f"(b) Citra Rusak (Mixed Noise)\nPSNR: {psnr_rusak:.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(hasil_fft_saja, cmap='gray')
    plt.title(f"(c) Baseline: FFT-Saja\nPSNR: {psnr_fft:.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(hasil_cnn_saja, cmap='gray')
    plt.title(f"(d) Baseline: 'CNN-Saja' (Simulasi)\nPSNR: {psnr_cnn:.2f} dB")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(hasil_final_hibrida, cmap='gray')
    plt.title(f"(e) Model Hibrida (Usulan)\nPSNR: {psnr_hibrida:.2f} dB")
    plt.axis('off')
    
    # Kosongkan subplot terakhir
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.7, "Analisis:", fontsize=14, ha='center')
    plt.text(0.5, 0.2, "1. FFT-Saja (c) menghilangkan garis,\n   tapi 'noise' acak tersisa.\n"
                       "2. 'CNN-Saja' (d) gagal total pada\n   garis periodik (dianggap tekstur).\n"
                       "3. Hibrida (e) berhasil: FFT\n   menghilangkan garis, 'CNN'\n   menghilangkan sisa noise.", 
             ha='center', va='top', fontsize=12)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Jalankan skrip utama
if __name__ == "__main__":
    main()