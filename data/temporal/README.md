# Uji Validitas Temporal 2026

Dataset utama berasal dari Twitter 2018–2019. Untuk mengukur relevansinya,
kumpulkan **100–200 komentar publik terbaru** (TikTok, Instagram, YouTube, atau X)
yang relevan dengan ujaran kebencian/abusif, lalu isi `temporal_2026.csv`.

## Format

```csv
text,label,annotator_1,annotator_2
"contoh teks",Normal,Normal,Normal
```

- `text` — teks asli (boleh sudah dianonimkan).
- `label` — label final hasil kesepakatan.
- `annotator_1`, `annotator_2` — label masing-masing anotator (opsional, untuk
  menghitung Cohen's kappa).

## Aturan anotasi

1. **Normal** — tidak mengandung ujaran kebencian maupun bahasa abusif.
2. **Abusive** — kata kasar/merendahkan, tetapi tidak menyerang kelompok.
3. **Hate Speech** — menyerang individu/kelompok berdasarkan agama, ras,
   etnis, gender, disabilitas, atau identitas lain.

Anotasi dilakukan independen oleh dua orang, lalu didiskusikan. Jangan
memasukkan data pribadi (nama, nomor telepon, alamat) ke CSV.

## Menjalankan

```bash
python src/evaluate_temporal.py
```

Output: `results/temporal_metrics.json` dan `results/temporal_predictions.csv`.
