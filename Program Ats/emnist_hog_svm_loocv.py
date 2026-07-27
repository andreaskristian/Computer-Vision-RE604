import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif (aman untuk server)

from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

# ===========================================================
#  KONFIGURASI PARAMETER
#  Sesuaikan parameter di sini untuk tuning
# ===========================================================

# --- HOG Parameters ---
HOG_PARAMS = {
    'orientations': 9,          # Jumlah bin orientasi (biasanya 6, 8, atau 9)
    'pixels_per_cell': (4, 4),  # Ukuran sel dalam piksel. (4,4) lebih detail dari (8,8)
    'cells_per_block': (2, 2),  # Jumlah sel per blok untuk normalisasi
    'block_norm': 'L2-Hys',     # Metode normalisasi blok ('L1','L2','L2-Hys')
    'visualize': False,
    'feature_vector': True
}

# --- SVM Parameters ---
SVM_PARAMS = {
    'kernel': 'rbf',     # Kernel: 'linear', 'rbf', 'poly', 'sigmoid'
    'C': 10.0,           # Parameter regularisasi. Lebih besar = lebih fit ke data
    'gamma': 'scale',    # Koefisien kernel untuk 'rbf','poly','sigmoid'
                         # 'scale' = 1/(n_features * X.var())
    'decision_function_shape': 'ovr',  # One-vs-Rest untuk multiclass
    'random_state': 42
}

# --- Dataset Settings ---
DATASET_SETTINGS = {
    # Path ke file CSV EMNIST
    # Program akan mencari di folder 'archive' terlebih dahulu (hasil download Kaggle)
    # lalu di folder yang sama dengan script ini
    'possible_csv_files': [
        'archive/emnist-letters-train.csv',       # ← Prioritas utama (folder archive)
        'archive/emnist-balanced-train.csv',
        'archive/emnist-digits-train.csv',
        'archive/emnist-byclass-train.csv',
        'emnist-letters-train.csv',               # ← Fallback (folder yang sama)
        'emnist-balanced-train.csv',
        'emnist-digits-train.csv',
        'emnist-byclass-train.csv',
    ],
    
    # Jumlah sampel untuk LOOCV (lebih banyak = lebih akurat tapi lebih lama)
    # Rekomendasi: 200-500 untuk tugas, 1000+ untuk performa maksimal
    # LOOCV dengan 500 sampel ≈ 1-3 menit
    'n_samples': 500,
    
    # Ukuran gambar EMNIST (selalu 28x28)
    'image_size': (28, 28),
    
    # Seed untuk reprodusibilitas
    'random_state': 42
}


# ===========================================================
#  FUNGSI-FUNGSI UTAMA
# ===========================================================

def print_header():
    """Cetak header program."""
    print("=" * 60)
    print("  EMNIST Classification: HOG + SVM + LOOCV")
    print("  Machine Vision (RE604) - Teknik Robotika")
    print("=" * 60)
    print()


def load_emnist_data(settings):
    """
    Memuat data EMNIST dari file CSV.
    
    Format CSV EMNIST:
        Kolom pertama = label (kelas)
        Kolom berikutnya = nilai piksel (784 piksel = 28x28)
    
    Returns:
        X: array gambar (n_samples, 28, 28)
        y: array label (n_samples,)
    """
    print("[1/5] Memuat Dataset EMNIST...")
    
    csv_path = None
    
    # Cari file CSV EMNIST
    for filename in settings['possible_csv_files']:
        if os.path.exists(filename):
            csv_path = filename
            break
    
    # Jika tidak ditemukan di folder saat ini, cek subfolder
    if csv_path is None:
        for root, dirs, files in os.walk('.'):
            for f in files:
                if any(name in f for name in ['emnist', 'EMNIST']) and f.endswith('.csv'):
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break
    
    # Jika masih tidak ditemukan, buat data sintetis untuk demonstrasi
    if csv_path is None:
        print("  [!] File CSV EMNIST tidak ditemukan.")
        print("  [!] Menggunakan data SINTETIS untuk demonstrasi.")
        print("  [!] Untuk hasil nyata, download dataset dari:")
        print("      https://www.kaggle.com/datasets/crawford/emnist/data")
        print()
        return generate_synthetic_data(settings)
    
    print(f"  File ditemukan: {csv_path}")
    print(f"  Membaca CSV... (ini mungkin memakan waktu)")
    
    # Baca CSV dengan efisien
    n_samples = settings['n_samples']
    df = pd.read_csv(csv_path, header=None, nrows=n_samples * 10)  # Baca lebih banyak untuk sampling
    
    y_raw = df.iloc[:, 0].values
    X_raw = df.iloc[:, 1:].values.astype(np.float32)
    
    # Ambil sampel terdistribusi merata per kelas
    X, y = stratified_sample(X_raw, y_raw, n_samples, settings['random_state'])
    
    # Reshape ke gambar 28x28
    X = X.reshape(-1, 28, 28)
    
    # EMNIST perlu dirotasi/di-flip karena format penyimpanannya
    X = np.array([np.fliplr(np.rot90(img, k=3)) for img in X])
    
    print(f"  Total sampel dimuat: {len(X)}")
    print(f"  Jumlah kelas: {len(np.unique(y))}")
    print(f"  Distribusi label: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()
    
    return X, y


def generate_synthetic_data(settings):
    """
    Generate data sintetis berbentuk karakter sederhana untuk demonstrasi.
    Digunakan jika dataset EMNIST tidak tersedia.
    """
    np.random.seed(settings['random_state'])
    n = settings['n_samples']
    n_classes = 5  # 5 kelas sintetis: 0, 1, 2, 3, 4
    
    X = []
    y = []
    
    per_class = n // n_classes
    
    for cls in range(n_classes):
        for _ in range(per_class):
            img = np.zeros((28, 28), dtype=np.float32)
            
            # Buat pola berbeda untuk setiap kelas
            if cls == 0:  # Lingkaran
                for i in range(28):
                    for j in range(28):
                        if 8 < ((i-14)**2 + (j-14)**2)**0.5 < 12:
                            img[i, j] = 255
            elif cls == 1:  # Garis vertikal
                img[4:24, 12:16] = 255
            elif cls == 2:  # Huruf L
                img[4:24, 8:12] = 255
                img[20:24, 8:20] = 255
            elif cls == 3:  # Huruf T
                img[4:8, 6:22] = 255
                img[4:22, 12:16] = 255
            elif cls == 4:  # Kotak
                img[6:22, 6:10] = 255
                img[6:22, 18:22] = 255
                img[6:10, 6:22] = 255
                img[18:22, 6:22] = 255
            
            # Tambah noise
            noise = np.random.normal(0, 15, (28, 28))
            img = np.clip(img + noise, 0, 255)
            X.append(img)
            y.append(cls)
    
    return np.array(X), np.array(y)


def stratified_sample(X, y, n_samples, random_state=42):
    """Ambil sampel terdistribusi merata per kelas."""
    np.random.seed(random_state)
    classes = np.unique(y)
    per_class = max(1, n_samples // len(classes))
    
    X_out, y_out = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, min(per_class, len(idx)), replace=False)
        X_out.append(X[chosen])
        y_out.append(y[chosen])
    
    return np.vstack(X_out), np.concatenate(y_out)


def extract_hog_features(X, hog_params):
    """
    Ekstraksi fitur HOG dari sekumpulan gambar.
    
    HOG (Histogram of Oriented Gradients):
    - Menghitung distribusi gradien (tepi/arah) dalam sel-sel lokal
    - Sangat efektif untuk mendeteksi bentuk/struktur karakter
    
    Args:
        X: array gambar (n, H, W), nilai 0-255
        hog_params: dict parameter HOG
    
    Returns:
        features: array (n, n_features)
    """
    print("[2/5] Ekstraksi Fitur HOG...")
    
    features = []
    for i, img in enumerate(X):
        # Normalisasi piksel ke 0-1
        img_norm = img / 255.0
        
        # Ekstraksi HOG
        feat = hog(
            img_norm,
            orientations=hog_params['orientations'],
            pixels_per_cell=hog_params['pixels_per_cell'],
            cells_per_block=hog_params['cells_per_block'],
            block_norm=hog_params['block_norm'],
            visualize=hog_params['visualize'],
            feature_vector=hog_params['feature_vector']
        )
        features.append(feat)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(X)} gambar diproses")
    
    features = np.array(features)
    print(f"  Dimensi fitur HOG per gambar: {features.shape[1]}")
    print(f"  Total dataset fitur: {features.shape}")
    print()
    
    return features


def normalize_features(X_features):
    """
    Normalisasi fitur menggunakan StandardScaler (z-score normalization).
    Penting untuk SVM agar skala fitur seragam.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    return X_scaled, scaler


def run_loocv(X_scaled, y, svm_params):
    """
    Evaluasi model menggunakan Leave-One-Out Cross-Validation (LOOCV).
    
    LOOCV:
    - Setiap iterasi: 1 sampel sebagai test, sisanya sebagai training
    - Total iterasi = jumlah sampel
    - Tidak ada bias pemilihan fold → estimasi akurasi paling tidak bias
    - Komputasi mahal untuk dataset besar
    
    Returns:
        y_pred: prediksi untuk seluruh dataset
        waktu_total: waktu eksekusi LOOCV
    """
    print("[3/5] Menjalankan Leave-One-Out Cross-Validation (LOOCV)...")
    print(f"  Total iterasi: {len(X_scaled)} (1 per sampel)")
    print(f"  Estimasi waktu: {len(X_scaled) * 0.05:.0f}-{len(X_scaled) * 0.2:.0f} detik")
    print()
    
    clf = SVC(**svm_params)
    loo = LeaveOneOut()
    
    start_time = time.time()
    
    # cross_val_predict dengan LOO - lebih efisien dari loop manual
    y_pred = cross_val_predict(clf, X_scaled, y, cv=loo, n_jobs=-1, verbose=0)
    
    waktu_total = time.time() - start_time
    
    print(f"  LOOCV selesai dalam {waktu_total:.2f} detik")
    print()
    
    return y_pred, waktu_total


def compute_metrics(y_true, y_pred):
    """
    Hitung semua metrik evaluasi:
    - Confusion Matrix
    - Accuracy
    - Precision (per kelas dan rata-rata)
    - F1-Score (per kelas dan rata-rata)
    """
    print("[4/5] Menghitung Metrik Evaluasi...")
    
    # Metrik utama
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }
    
    print()
    print("=" * 60)
    print("  HASIL EVALUASI LOOCV")
    print("=" * 60)
    print(f"  Accuracy          : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (macro) : {precision_macro:.4f}")
    print(f"  Precision (wtd)   : {precision_weighted:.4f}")
    print(f"  F1-Score (macro)  : {f1_macro:.4f}")
    print(f"  F1-Score (wtd)    : {f1_weighted:.4f}")
    print("=" * 60)
    print()
    print("Classification Report per Kelas:")
    print(metrics['classification_report'])
    
    return metrics


def visualize_results(X, y_true, y_pred, metrics, hog_params, svm_params):
    """
    Visualisasi hasil: confusion matrix, sampel gambar, dan fitur HOG.
    Menyimpan gambar ke file PNG.
    """
    print("[5/5] Membuat Visualisasi...")
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # -------------------------------------------------------
    # GAMBAR 1: Confusion Matrix
    # -------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(max(8, n_classes), max(6, n_classes)))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics['confusion_matrix'],
        display_labels=classes
    )
    disp.plot(ax=ax1, cmap='Blues', colorbar=True)
    
    ax1.set_title(
        f"Confusion Matrix - LOOCV\n"
        f"HOG+SVM | Accuracy: {metrics['accuracy']*100:.2f}% | "
        f"F1: {metrics['f1_macro']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Disimpan: confusion_matrix.png")
    
    # -------------------------------------------------------
    # GAMBAR 2: Contoh Gambar + Visualisasi HOG
    # -------------------------------------------------------
    n_show = min(5, len(X))
    fig2, axes = plt.subplots(n_show, 3, figsize=(10, n_show * 2.5))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        img = X[i] / 255.0
        
        # HOG visualization
        _, hog_image = hog(
            img,
            orientations=hog_params['orientations'],
            pixels_per_cell=hog_params['pixels_per_cell'],
            cells_per_block=hog_params['cells_per_block'],
            block_norm=hog_params['block_norm'],
            visualize=True,
            feature_vector=True
        )
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        # Gambar asli
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Asli\nLabel: {y_true[i]}", fontsize=9)
        axes[i, 0].axis('off')
        
        # HOG features
        axes[i, 1].imshow(hog_image_rescaled, cmap='gray')
        axes[i, 1].set_title("HOG Features", fontsize=9)
        axes[i, 1].axis('off')
        
        # Status prediksi
        status = "✓ Benar" if y_true[i] == y_pred[i] else f"✗ Salah\n(pred: {y_pred[i]})"
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        axes[i, 2].text(0.5, 0.5, status, ha='center', va='center',
                        fontsize=12, color=color, fontweight='bold',
                        transform=axes[i, 2].transAxes)
        axes[i, 2].set_title("Hasil Prediksi", fontsize=9)
        axes[i, 2].axis('off')
    
    fig2.suptitle(
        "Contoh Gambar, HOG Features, dan Hasil Prediksi\n"
        f"HOG: orient={hog_params['orientations']}, "
        f"ppc={hog_params['pixels_per_cell']}, "
        f"cpb={hog_params['cells_per_block']}",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('hog_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Disimpan: hog_visualization.png")
    
    # -------------------------------------------------------
    # GAMBAR 3: Ringkasan Metrik
    # -------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart metrik
    metric_names = ['Accuracy', 'Precision\n(Macro)', 'Precision\n(Weighted)',
                    'F1-Score\n(Macro)', 'F1-Score\n(Weighted)']
    metric_values = [
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['precision_weighted'],
        metrics['f1_macro'],
        metrics['f1_weighted']
    ]
    
    colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#FF5722']
    bars = axes3[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=0.5)
    axes3[0].set_ylim(0, 1.1)
    axes3[0].set_ylabel('Score', fontsize=11)
    axes3[0].set_title('Ringkasan Metrik Evaluasi (LOOCV)', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars, metric_values):
        axes3[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Info parameter
    param_text = (
        f"=== HOG Parameters ===\n"
        f"Orientations  : {hog_params['orientations']}\n"
        f"Pixels/Cell   : {hog_params['pixels_per_cell']}\n"
        f"Cells/Block   : {hog_params['cells_per_block']}\n"
        f"Block Norm    : {hog_params['block_norm']}\n\n"
        f"=== SVM Parameters ===\n"
        f"Kernel        : {svm_params['kernel']}\n"
        f"C             : {svm_params['C']}\n"
        f"Gamma         : {svm_params['gamma']}\n\n"
        f"=== Evaluasi ===\n"
        f"Metode        : LOOCV\n"
        f"N Sampel      : {len(y_true)}\n"
        f"N Kelas       : {n_classes}"
    )
    
    axes3[1].text(0.05, 0.95, param_text, transform=axes3[1].transAxes,
                  fontsize=10, va='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes3[1].axis('off')
    axes3[1].set_title('Konfigurasi Parameter', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Disimpan: metrics_summary.png")
    print()


def save_results_txt(metrics, hog_params, svm_params, n_samples, waktu):
    """Simpan hasil evaluasi ke file teks."""
    with open('hasil_evaluasi.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  HASIL EVALUASI: HOG + SVM + LOOCV\n")
        f.write("  Machine Vision (RE604) - Teknik Robotika\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[HOG Parameters]\n")
        for k, v in hog_params.items():
            f.write(f"  {k:20s}: {v}\n")
        
        f.write("\n[SVM Parameters]\n")
        for k, v in svm_params.items():
            f.write(f"  {k:20s}: {v}\n")
        
        f.write(f"\n[Evaluasi]\n")
        f.write(f"  {'Metode':20s}: Leave-One-Out Cross-Validation (LOOCV)\n")
        f.write(f"  {'N Sampel':20s}: {n_samples}\n")
        f.write(f"  {'Waktu LOOCV':20s}: {waktu:.2f} detik\n")
        
        f.write(f"\n[Metrik Evaluasi]\n")
        f.write(f"  {'Accuracy':20s}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  {'Precision (macro)':20s}: {metrics['precision_macro']:.4f}\n")
        f.write(f"  {'Precision (weighted)':20s}: {metrics['precision_weighted']:.4f}\n")
        f.write(f"  {'F1-Score (macro)':20s}: {metrics['f1_macro']:.4f}\n")
        f.write(f"  {'F1-Score (weighted)':20s}: {metrics['f1_weighted']:.4f}\n")
        
        f.write("\n[Classification Report]\n")
        f.write(metrics['classification_report'])
        
        f.write("\n[Confusion Matrix]\n")
        f.write(str(metrics['confusion_matrix']))
    
    print("  Disimpan: hasil_evaluasi.txt")


# ===========================================================
#  MAIN
# ===========================================================

def main():
    print_header()
    
    # 1. Load Data
    X, y = load_emnist_data(DATASET_SETTINGS)
    
    # Encode label jika berupa string
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Ekstraksi Fitur HOG
    X_features = extract_hog_features(X, HOG_PARAMS)
    
    # 3. Normalisasi Fitur
    X_scaled, scaler = normalize_features(X_features)
    print(f"  Fitur dinormalisasi (StandardScaler)")
    print()
    
    # 4. LOOCV
    y_pred_encoded, waktu = run_loocv(X_scaled, y_encoded, SVM_PARAMS)
    
    # Decode kembali ke label asli
    y_pred = le.inverse_transform(y_pred_encoded)
    y_true = le.inverse_transform(y_encoded)
    
    # 5. Hitung Metrik
    metrics = compute_metrics(y_true, y_pred)
    
    # 6. Visualisasi
    visualize_results(X, y_true, y_pred, metrics, HOG_PARAMS, SVM_PARAMS)
    
    # 7. Simpan ke file teks
    save_results_txt(metrics, HOG_PARAMS, SVM_PARAMS, len(X), waktu)
    
    print("=" * 60)
    print("  SELESAI! File output:")
    print("  - confusion_matrix.png   → Confusion Matrix")
    print("  - hog_visualization.png  → Contoh HOG Features")
    print("  - metrics_summary.png    → Ringkasan Metrik")
    print("  - hasil_evaluasi.txt     → Laporan Lengkap")
    print("=" * 60)


if __name__ == "__main__":
    main()