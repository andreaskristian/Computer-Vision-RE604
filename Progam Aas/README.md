# OCR Plat Nomor Kendaraan menggunakan VLM (LM Studio) + Python

Proyek ini melakukan Optical Character Recognition (OCR) pada plat nomor
kendaraan Indonesia menggunakan Visual Language Model (VLM) multimodal
(mis. LLaVA / BakLLaVA) yang dijalankan secara lokal melalui **LM Studio**,
lalu dievaluasi menggunakan metrik **Character Error Rate (CER)**.

Tugas: Asesmen Akhir Semester — Computer Vision (RE604), Teknik Robotika.

---

## 1. Struktur Folder

```
vlm_ocr_project/
├── ocr_vlm_lmstudio.py       # Script utama: kirim gambar ke LM Studio, simpan CSV, hitung CER
├── cer_utils.py               # Implementasi rumus CER = (S + D + I) / N
├── prepare_ground_truth.py    # Helper: buat template ground_truth.csv & crop plat (opsional)
├── requirements.txt
└── README.md
```

## 2. Persiapan Server VLM Lokal

Ada 2 opsi, tergantung apakah CPU mendukung **AVX2** atau tidak.

### Opsi A — LM Studio (jika CPU mendukung AVX2)

1. Download & install LM Studio: https://lmstudio.ai
2. Di dalam LM Studio, buka tab **Search**, download model multimodal, misalnya:
   - `llava-v1.5-7b`, `bakllava`, atau `Qwen2-VL-2B-Instruct`
3. Buka tab **Local Server** (ikon `<->`), pilih model yang sudah didownload,
   klik **Start Server**. Default berjalan di `http://localhost:1234`.
4. Catat nama model persis seperti yang tertulis di LM Studio.
5. Jalankan script dengan `--base_url http://localhost:1234/v1/chat/completions`.

Referensi resmi: https://lmstudio.ai/docs/python/llm-prediction/image-input

### Opsi B — KoboldCPP (untuk CPU lama tanpa AVX2, mis. Ivy Bridge/Sandy Bridge)

LM Studio versi terbaru mewajibkan CPU dengan AVX2. Untuk CPU lama, gunakan
**KoboldCPP** build khusus CPU lama (masih memakai model GGUF yang sama):

1. Download model GGUF vision (mis. `Qwen2-VL-2B-Instruct`) + file
   **mmproj**-nya dari HuggingFace (bisa lewat fitur Search di LM Studio
   walau LM Studio-nya sendiri tidak dipakai untuk inferensi — cukup untuk
   download filenya saja, lalu tetap dipakai lewat KoboldCPP).
2. Download KoboldCPP dari https://github.com/LostRuins/koboldcpp/releases
   — pilih **`koboldcpp-oldpc.exe`** (untuk Windows, CPU tanpa AVX2).
3. Jalankan `koboldcpp-oldpc.exe`. Di tab **Quick Launch**:
   - Backend: **Use CPU (Old CPU)**
   - GGUF Text Model: pilih file model utama (bukan mmproj)
4. Di tab **Loaded Files**: isi kolom **Vision mmproj** dengan file
   `mmproj-*.gguf`.
5. Klik **Launch**. Server default berjalan di `http://localhost:5001`.
6. Jalankan script dengan `--base_url http://localhost:5001/v1/chat/completions`
   (ini juga default project ini, jadi bisa tidak usah ditulis eksplisit).

> Catatan: karena berjalan full-CPU (tanpa akselerasi GPU), inferensi per
> gambar bisa memakan waktu puluhan detik hingga beberapa menit — normal
> untuk hardware lama. Tetap dijelaskan di README/video bahwa konsep VLM
> untuk OCR yang digunakan sama; hanya tool eksekusi lokal yang menyesuaikan
> keterbatasan hardware.

## 3. Persiapan Dataset

1. Download dataset dari Kaggle:
   https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
2. Gunakan folder **test** saja sesuai instruksi soal.
3. Dataset ini pada umumnya berformat YOLO (gambar + label bounding box),
   **tanpa** teks plat nomor sebagai ground truth. Karena itu, ground truth
   harus dibuat/dibaca manual oleh mahasiswa dari tiap gambar. Gunakan
   helper di bawah untuk mempermudah:

```bash
# Buat template CSV kosong dari semua gambar di folder test
python prepare_ground_truth.py generate \
    --images_dir dataset/test/images \
    --output_csv ground_truth.csv
```

Lalu buka `ground_truth.csv` dan isi kolom `ground_truth` dengan membaca
plat nomor pada tiap gambar (contoh isi: `BP 1234 CD`).

**(Opsional, disarankan)** Jika folder test memiliki `labels/` berisi
bounding box YOLO, crop dulu area plat nomornya supaya VLM lebih akurat
membaca teks (foto full-frame mobil biasanya kurang jelas untuk OCR):

```bash
python prepare_ground_truth.py crop \
    --images_dir dataset/test/images \
    --labels_dir dataset/test/labels \
    --output_dir dataset/test/cropped
```

Jika kamu crop, gunakan `dataset/test/cropped` sebagai `--images_dir` pada
langkah berikutnya.

## 4. Instalasi

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Menjalankan OCR + Evaluasi CER

Pastikan LM Studio Local Server sudah berjalan (langkah 2), lalu:

```bash
# Default project ini sudah diarahkan ke KoboldCPP (localhost:5001)
python ocr_vlm_lmstudio.py \
    --images_dir dataset/test/cropped \
    --ground_truth_csv ground_truth.csv \
    --output_csv results.csv \
    --model koboldcpp/Qwen2-VL-2B-Instruct

# Kalau pakai LM Studio, tambahkan --base_url:
python ocr_vlm_lmstudio.py \
    --images_dir dataset/test/cropped \
    --ground_truth_csv ground_truth.csv \
    --base_url http://localhost:1234/v1/chat/completions \
    --model llava-v1.5-7b
```

Parameter:
| Argumen | Wajib | Keterangan |
|---|---|---|
| `--images_dir` | ya | Folder gambar yang akan di-OCR |
| `--ground_truth_csv` | ya | CSV kolom `image, ground_truth` |
| `--output_csv` | tidak (default `results.csv`) | File hasil |
| `--model` | tidak | Nama model (bebas untuk KoboldCPP, harus persis untuk LM Studio) |
| `--base_url` | tidak (default KoboldCPP `:5001`) | Endpoint server VLM lokal |
| `--limit` | tidak | Batasi jumlah gambar (untuk uji coba cepat) |

Output `results.csv` berisi kolom: `image, ground_truth, prediction, CER_score`,
dan rata-rata CER dicetak di akhir eksekusi.

## 6. Rumus CER

```
CER = (S + D + I) / N
```
- S = jumlah karakter substitusi
- D = jumlah karakter yang dihapus (ada di ground truth, hilang di prediksi)
- I = jumlah karakter yang disisipkan (ada di prediksi, tidak ada di ground truth)
- N = jumlah karakter pada ground truth

Dihitung via Levenshtein edit-distance dengan backtrace (lihat `cer_utils.py`).
CER = 0 berarti prediksi sempurna; semakin besar CER, semakin banyak kesalahan.

## 7. Troubleshooting

- **Connection refused / timeout**: pastikan LM Studio Local Server sudah
  running dan port `1234` tidak diblok firewall.
- **Model tidak merespons gambar**: pastikan model yang dipilih memang
  mendukung vision/multimodal (bukan model teks biasa).
- **Prediksi berisi teks tambahan** (bukan hanya plat nomor): fungsi
  `clean_prediction()` di `ocr_vlm_lmstudio.py` sudah menormalisasi output,
  tapi bisa disesuaikan lagi jika model sering menjawab dengan kalimat penuh.

## 8. Catatan untuk Video Penjelasan (Soal No. 2)

Video (maks. 10 menit, wajib tampil pembicara) sebaiknya mencakup:
1. **Konsep VLM & penerapannya untuk OCR** — jelaskan singkat apa itu VLM,
   bedanya dengan OCR klasik (Tesseract/CRNN), kenapa VLM bisa membaca
   teks dari gambar tanpa training khusus.
2. **Cara kerja integrasi LM Studio & Python** — tunjukkan Local Server LM
   Studio berjalan, lalu tunjukkan kode `query_lmstudio()` yang mengirim
   gambar (base64) + prompt via HTTP POST ke endpoint `/v1/chat/completions`.
3. **Proses inferensi & evaluasi** — jalankan `ocr_vlm_lmstudio.py` live atau
   tunjukkan hasil `results.csv`, jelaskan bagaimana CER dihitung.
4. **Contoh sukses vs gagal** — tampilkan 1-2 gambar dengan CER rendah
   (prediksi akurat) dan 1-2 gambar dengan CER tinggi, lalu analisis kenapa
   gagal (misalnya: plat blur, sudut kamera miring, karakter mirip seperti
   0/O atau 8/B, pencahayaan kurang).

## 9. Upload ke GitHub

```bash
git init
git add .
git commit -m "OCR plat nomor VLM LM Studio - Asesmen Computer Vision RE604"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Pastikan repository bersifat **public**, lalu submit link GitHub melalui e-learning.
