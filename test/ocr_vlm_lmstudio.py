"""
ocr_vlm_lmstudio.py
--------------------
OCR plat nomor kendaraan menggunakan Visual Language Model (VLM) lokal
(mis. Qwen2-VL) yang dijalankan via server lokal ber-API OpenAI-compatible
-- bisa LM Studio ATAU KoboldCPP -- diintegrasikan dengan Python.

Alur:
  1. Baca daftar gambar + ground truth dari ground_truth.csv
  2. Kirim tiap gambar ke server VLM lokal dengan prompt:
     "What is the license plate number shown in this image?
     Respond only with the plate number."
  3. Simpan hasil ke CSV: image, ground_truth, prediction, CER_score
  4. Hitung rata-rata CER di akhir

Prasyarat:
  - Server VLM lokal sudah jalan:
      * LM Studio  -> Local Server, default http://localhost:1234
      * KoboldCPP   -> default http://localhost:5001
    (KoboldCPP dipakai sebagai alternatif untuk CPU lama yang tidak
    mendukung AVX2, menggunakan backend "Use CPU (Old CPU)")
  - `pip install -r requirements.txt`

Contoh penggunaan (KoboldCPP, default project ini):
  python ocr_vlm_lmstudio.py \
      --images_dir dataset/test/images \
      --ground_truth_csv ground_truth.csv \
      --output_csv results.csv \
      --model koboldcpp/Qwen2-VL-2B-Instruct \
      --base_url http://localhost:5001/v1/chat/completions

Contoh penggunaan (LM Studio):
  python ocr_vlm_lmstudio.py \
      --images_dir dataset/test/images \
      --ground_truth_csv ground_truth.csv \
      --base_url http://localhost:1234/v1/chat/completions \
      --model llava-v1.5-7b
"""

import os
import csv
import base64
import argparse
import time
from pathlib import Path

import requests

from cer_utils import compute_cer

DEFAULT_BASE_URL = "http://localhost:5001/v1/chat/completions"  # KoboldCPP default
PROMPT = (
    "What is the license plate number shown in this image? "
    "Respond only with the plate number."
)


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext in ("png", "webp", "bmp", "gif"):
        return ext
    return "jpeg"


def query_vlm(image_path: str, model_name: str, base_url: str, timeout: int = 180) -> str:
    """Kirim satu gambar ke server VLM lokal (LM Studio / KoboldCPP) dan
    kembalikan teks jawaban model. Kedua server ini menyediakan endpoint
    /v1/chat/completions yang kompatibel dengan format OpenAI."""
    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    resp = requests.post(base_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def clean_prediction(text: str) -> str:
    """Normalisasi output model: uppercase, buang newline/tanda baca berlebih."""
    if not text:
        return ""
    text = text.strip().upper()
    text = text.replace("\n", " ").replace(".", "").replace('"', "")
    return " ".join(text.split())


def load_ground_truth(csv_path: str) -> dict:
    gt_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row["image"].strip()
            gt = row["ground_truth"].strip().upper()
            if image:
                gt_map[image] = gt
    return gt_map


def main():
    parser = argparse.ArgumentParser(description="OCR plat nomor via VLM (LM Studio) + evaluasi CER")
    parser.add_argument("--images_dir", required=True, help="Folder berisi gambar test")
    parser.add_argument("--ground_truth_csv", required=True,
                         help="CSV dengan kolom: image, ground_truth")
    parser.add_argument("--output_csv", default="results.csv", help="Path output CSV hasil")
    parser.add_argument("--model", default="koboldcpp/model",
                         help="Nama model. Untuk KoboldCPP bebas (server hanya punya 1 model "
                              "yang di-load), untuk LM Studio harus persis nama di Local Server")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL,
                         help="Endpoint chat completions. KoboldCPP: http://localhost:5001/v1/chat/completions "
                              "| LM Studio: http://localhost:1234/v1/chat/completions")
    parser.add_argument("--limit", type=int, default=None,
                         help="Batasi jumlah gambar diproses (untuk uji coba cepat)")
    args = parser.parse_args()

    gt_map = load_ground_truth(args.ground_truth_csv)
    items = list(gt_map.items())
    if args.limit:
        items = items[: args.limit]

    results = []
    total_cer = 0.0
    n_ok = 0

    for idx, (image_name, gt_text) in enumerate(items, start=1):
        image_path = os.path.join(args.images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[{idx}/{len(items)}] [SKIP] file tidak ditemukan: {image_path}")
            continue

        try:
            raw_pred = query_vlm(image_path, args.model, args.base_url)
        except Exception as e:
            print(f"[{idx}/{len(items)}] [ERROR] {image_name}: {e}")
            raw_pred = ""

        pred = clean_prediction(raw_pred)
        cer, S, D, I = compute_cer(gt_text, pred)

        results.append([image_name, gt_text, pred, round(cer, 4)])
        total_cer += cer
        n_ok += 1

        print(f"[{idx}/{len(items)}] {image_name} | GT='{gt_text}' PRED='{pred}' "
              f"CER={cer:.4f} (S={S} D={D} I={I})")

        time.sleep(0.05)  # jeda kecil, hindari membanjiri local server

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth", "prediction", "CER_score"])
        writer.writerows(results)

    print("\n============================")
    if n_ok:
        print(f"Rata-rata CER: {total_cer / n_ok:.4f}  ({n_ok} gambar diproses)")
    print(f"Hasil disimpan di: {args.output_csv}")
    print("============================")


if __name__ == "__main__":
    main()
