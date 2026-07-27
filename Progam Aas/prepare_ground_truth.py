"""
prepare_ground_truth.py
------------------------
Dataset "Indonesian License Plate Dataset" di Kaggle berformat YOLO
(gambar + file label .txt berisi bounding box), TANPA teks plat nomor.
Satu foto bisa memuat LEBIH DARI SATU plat (lebih dari 1 baris di file
label -- misal foto jalanan dengan beberapa mobil). Karena itu, ground
truth teks harus diisi manual oleh mahasiswa setelah tiap plat di-crop
jadi gambar tersendiri.

Script ini membantu 2 hal:
  1. generate  -> membuat template ground_truth.csv (kolom image + ground_truth
                  kosong) dari semua file gambar di folder test, supaya tinggal
                  diisi manual satu per satu. (Gunakan HANYA jika tidak ada
                  label YOLO / tidak melakukan crop -- 1 baris per foto.)
  2. crop      -> membaca label YOLO (class x_center y_center width height,
                  normalized 0-1), meng-crop SETIAP bounding box di tiap foto
                  jadi file gambar terpisah (mis. test001_0.jpg, test001_1.jpg
                  untuk 2 plat dalam 1 foto test001.jpg). VLM akan jauh lebih
                  akurat membaca teks pada plat yang sudah di-crop close-up
                  dibanding foto jalanan full-frame. Bisa langsung sekaligus
                  membuat template ground_truth.csv (1 baris per plat) dengan
                  --ground_truth_template.

Contoh:
  python prepare_ground_truth.py generate --images_dir dataset/test/images \
      --output_csv ground_truth.csv

  python prepare_ground_truth.py crop --images_dir dataset/test/images \
      --labels_dir dataset/test/labels --output_dir dataset/test/cropped \
      --ground_truth_template ground_truth.csv
"""

import os
import csv
import argparse

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def cmd_generate(args):
    files = sorted(
        f for f in os.listdir(args.images_dir) if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        print(f"Tidak ada gambar ditemukan di {args.images_dir}")
        return

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth"])
        for fname in files:
            writer.writerow([fname, ""])

    print(f"Template dibuat: {args.output_csv} ({len(files)} baris)")
    print("Silakan buka file ini dan isi kolom 'ground_truth' dengan membaca "
          "plat nomor pada tiap gambar secara manual.")


def yolo_to_box(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, w, h = parts[:5]
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)


def cmd_crop(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(args.images_dir) if f.lower().endswith(IMAGE_EXTS)
    )
    n_cropped = 0
    crop_names = []  # untuk dipakai generate ground_truth.csv otomatis

    for fname in files:
        stem = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        label_path = os.path.join(args.labels_dir, stem + ".txt")
        img_path = os.path.join(args.images_dir, fname)
        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(label_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        if not lines:
            continue

        # Satu foto bisa berisi lebih dari satu plat (lebih dari 1 baris label).
        # Setiap plat di-crop jadi file terpisah: <stem>_0.jpg, <stem>_1.jpg, dst.
        for idx, line in enumerate(lines):
            box = yolo_to_box(line, w, h)
            if box is None:
                continue
            cropped = img.crop(box)
            out_name = f"{stem}_{idx}{ext}"
            out_path = os.path.join(args.output_dir, out_name)
            cropped.save(out_path)
            crop_names.append(out_name)
            n_cropped += 1

    print(f"Selesai crop: {n_cropped} plat (dari {len(files)} foto) disimpan di {args.output_dir}")

    if args.ground_truth_template:
        with open(args.ground_truth_template, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "ground_truth"])
            for name in crop_names:
                writer.writerow([name, ""])
        print(f"Template ground truth dibuat: {args.ground_truth_template} "
              f"({len(crop_names)} baris, satu per plat hasil crop)")
        print("Silakan isi kolom 'ground_truth' dengan membaca tiap gambar hasil crop.")


def main():
    parser = argparse.ArgumentParser(description="Helper: siapkan ground truth / crop plat dari label YOLO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Buat template ground_truth.csv")
    p_gen.add_argument("--images_dir", required=True)
    p_gen.add_argument("--output_csv", default="ground_truth.csv")
    p_gen.set_defaults(func=cmd_generate)

    p_crop = sub.add_parser("crop", help="Crop area plat nomor dari label YOLO")
    p_crop.add_argument("--images_dir", required=True)
    p_crop.add_argument("--labels_dir", required=True)
    p_crop.add_argument("--output_dir", required=True)
    p_crop.add_argument("--ground_truth_template", default=None,
                         help="Jika diisi, otomatis buat template ground_truth.csv "
                              "(1 baris per plat hasil crop)")
    p_crop.set_defaults(func=cmd_crop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
