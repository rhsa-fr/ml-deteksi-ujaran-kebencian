# Outline Presentasi — 14 Slide

Deck siap pakai: `reports/presentasi_tugas_besar.pptx` (16:9, tema brutalist).
File ini adalah versi teks untuk diedit cepat.

## Slide 1 — Judul
- Deteksi Ujaran Kebencian dan Bahasa Abusif pada Teks Bahasa Indonesia
- Tugas Besar Kecerdasan Buatan 2026
- Kelompok, nama, NIM, tanggal

## Slide 2 — Masalah
- 13.169 tweet pada dataset 2018–2019.
- 3 kelas: Normal, Abusive, Hate Speech.
- Moderasi manual tidak skalabel; daftar kata gagal pada sarkasme/negasi.
- Pertanyaan: seberapa jauh model klasik sederhana bisa membantu?

## Slide 3 — Dataset
- Ibrohim & Budi (2019), 12 label asli.
- 13.169 tweet → 12.828 setelah dedup (341 duplikat).
- Normal 5.701 (44,44%), Abusive 1.709 (13,32%), Hate Speech 5.418 (42,24%).
- Pemetaan: HS=1 → Hate Speech; Abusive=1 tanpa HS → Abusive; sisanya Normal.

## Slide 4 — Pipeline
- 01 Dataset → 02 Praproses → 03 Split → 04 TF-IDF → 05 Model → 06 Evaluasi → 07 Demo.
- Setiap tahap: konsep → implementasi → hasil → analisis.
- Split 10.262 / 1.283 / 1.283.

## Slide 5 — Praproses
- Case folding, URL/mention, karakter berulang, non-alfabet.
- Normalisasi slang 15.151 entri + istilah 2026 (rizz, flexing, mager).
- Kata abusif tidak dinormalisasi; stopword tidak dihapus.
- Contoh sebelum/sesudah.

## Slide 6 — TF-IDF
- Word 1–2 gram: 15.315 fitur.
- char_wb 3–5 gram: 50.000 fitur.
- Total 65.315 fitur; `min_df=3`, `max_df=0,95`, `sublinear_tf=True`.

## Slide 7 — Model
- Multinomial NB: Bayes + independensi; α = 0,1.
- Regresi Logistik: cross-entropy + softmax; C = 1,0.
- Linear SVM: margin maksimum + hinge loss; C = 0,1.
- Tuning di validation; test sekali.

## Slide 8 — Hasil
| Model | Val F1 | Acc | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| NB | 0,7698 | 79,89% | 0,7601 | 0,8018 |
| LR | 0,8023 | 82,00% | 0,7859 | 0,8216 |
| SVM | 0,8056 | 83,71% | 0,8075 | 0,8365 |

- SVM menang tipis; data teks berdimensi tinggi cenderung linear-separable.

## Slide 9 — Confusion Matrix
- 1.074 benar / 209 salah dari 1.283.
- Hate Speech → Normal 70; Normal → Hate Speech 42.
- Abusive → Hate Speech 35; Hate Speech → Abusive 26.
- Kebingungan terbesar: Hate Speech ↔ Normal (112).

## Slide 10 — Per Kelas
- Normal F1 0,8729.
- Abusive F1 0,7122 (tersulit; hanya 13,3% data).
- Hate Speech F1 0,8376.
- Macro-F1 0,8075.

## Slide 11 — Error Analysis
- Hate → Normal: sarkasme tanpa kata kasar.
- Abusive → Hate: kata kasar + kata kelompok; batas kabur.
- Normal → Hate: topik politik/agama memicu false positive.
- Model bag-of-words tidak paham negasi, pelaku, target, konteks.

## Slide 12 — Demo
- Input teks bebas; label + confidence; skor tiga kelas.
- Token pendorong/penahan; teks setelah praproses.
- Endpoint: `/api/predict`, `/api/model-info`, `/api/health`, `/docs`.
- Jalankan: `uvicorn app.main:app --port 8000`.

## Slide 13 — Batasan
- Dataset 2019; bahasa/platform berubah.
- Domain shift belum diuji pada 2026 (template uji tersedia).
- Bias politik: cebong/ahok/prabowo menjadi pintasan.
- Bukan alat produksi; skor SVM belum terkalibrasi.

## Slide 14 — Kesimpulan
- Macro-F1 0,8075; akurasi 83,71%; 1.283 teks uji.
- Baseline kuat dan murah untuk penyaringan awal.
- Next: IndoBERT, uji temporal 2026, kalibrasi, fitur konteks/negasi.

## Catatan pembicara
- Jangan mengklaim model “memahami” bahasa; ia memakai pola leksikal.
- Tekankan pemilihan model di validation dan test yang hanya dipakai sekali.
- Saat demo, mulai dari contoh Normal, lalu Abusive, lalu Hate Speech.
- Siapkan jawaban: “kenapa bukan IndoBERT?” → biaya, baseline, rencana pengembangan.
- Siapkan jawaban: “datanya 2019?” → batasan + rencana uji temporal.
