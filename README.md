# Deteksi Ujaran Kebencian Bahasa Indonesia

Tugas Besar Kecerdasan Buatan 2026 — pipeline klasifikasi teks tiga kelas
(**Normal**, **Abusive**, **Hate Speech**) dengan TF-IDF dan tiga model klasik,
dilengkapi demo web.

## Ringkasan

- **Dataset:** Ibrohim & Budi (2019), 13.169 tweet bahasa Indonesia (12.828 setelah deduplikasi).
- **Fitur:** TF-IDF word 1–2 gram + char_wb 3–5 gram.
- **Model:** Multinomial Naive Bayes, Regresi Logistik, Linear SVM.
- **Evaluasi:** split stratified 80/10/10, tuning di validation, 5-fold CV,
  evaluasi akhir sekali di test (macro-F1 sebagai metrik utama).
- **Demo:** FastAPI + halaman HTML statis dengan label, skor kelas, dan kata
  yang paling memengaruhi prediksi.

## Struktur

```text
tugas-besar-nlp/
├─ app/main.py                # API FastAPI (entrypoint Vercel)
├─ public/index.html          # UI demo (static, dipromosikan ke CDN Vercel)
├─ data/
│  ├─ raw/                    # CSV mentah + kamus slang
│  ├─ processed/              # dataset bersih + ringkasan EDA
│  └─ temporal/               # template uji validitas temporal 2026
├─ models/                    # model .joblib + config praproses
├─ results/                   # metrik, confusion matrix, error analysis
├─ src/
│  ├─ download_data.py
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ inference.py
│  └─ evaluate_temporal.py
├─ reports/                   # laporan dan outline slide
├─ requirements.txt           # dependensi runtime (Vercel)
├─ requirements-dev.txt       # dependensi training + laporan
├─ vercel.json
└─ README.md
```

## Menjalankan

```bash
python -m pip install -r requirements-dev.txt

python src/download_data.py      # unduh dataset dari GitHub
python src/preprocess.py         # bersihkan + normalisasi + netralisasi entitas
python src/train.py              # model tiga kelas (studi pembanding)
python src/train_binary.py       # model final biner untuk demo
python src/threshold_analysis.py # opsional: analisis ambang indikasi hate

uvicorn app.main:app --reload --port 8000
```

Buka `http://localhost:8000`. Dokumentasi API otomatis di `http://localhost:8000/docs`.

## Deploy ke Vercel

Proyek memakai **FastAPI preset** Vercel: `app/main.py` sebagai entrypoint,
`public/` sebagai static CDN, dan `requirements.txt` sebagai dependensi runtime.

```bash
npm i -g vercel
vercel login
vercel          # preview
vercel --prod   # production
```

Atau import repo lewat dashboard Vercel:

| Setting | Nilai |
|---|---|
| Root Directory | `tugas-besar-nlp` |
| Framework Preset | `FastAPI` (auto-detect) |
| Build Command | kosong |
| Output Directory | kosong |
| Environment Variables | tidak ada |

Catatan deploy:

- `vercel.json` mengecualikan artefak training (`data/processed`, `models/pipeline_*.joblib`,
  `results`, `reports`) supaya bundle tetap kecil.
- Yang ikut bundle: `app/`, `src/`, `models/best_model.joblib`,
  `models/preprocess_config.json`, `data/raw/*.csv`, dan `public/index.html`.
- Cold start sekitar 1–3 detik karena scikit-learn dimuat; cocok untuk demo,
  bukan trafik tinggi.
- Batas bundle Python Vercel 500 MB; karena itu dependensi runtime dipisah
  dari `requirements-dev.txt`.
- Setelah deploy, cek `/api/health`, `/api/model-info`, dan `/docs`.

## Hasil

Angka diisi otomatis oleh `src/train.py` (tiga kelas) dan `src/train_binary.py`
(model final biner) ke `results/`.

**Skema biner — model final yang dipakai demo** (`src/train_binary.py`):

| Model | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Multinomial NB (α = 0,5) | 0,8379 | 86,05% | 0,8573 |
| Regresi Logistik (C = 5,0) | 0,8610 | 88,70% | 0,8860 |
| **Linear SVM (C = 0,1)** | **0,8635** | **89,01%** | **0,8893** |

Per kelas untuk SVM: Normal F1 0,8798; Kebencian/Abusif 0,8988. Pada 24 kalimat
probe bergaya 2026, akurasi 83,3% (20/24).

**Skema tiga kelas — studi pembanding** (`src/train.py`):

| Model | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Multinomial NB (α = 0,5) | 0,7268 | 81,76% | 0,7710 |
| Regresi Logistik (C = 5,0) | 0,7444 | 84,96% | 0,8189 |
| **Linear SVM (C = 0,1)** | **0,7490** | 84,57% | 0,8181 |

Per kelas untuk SVM tiga kelas: Normal F1 0,8800; Abusive 0,7294;
Hate Speech 0,8449. Pada probe 2026 yang sama, akurasi hanya 58,3%.

Model final dipilih dari macro-F1 validation, lalu dievaluasi sekali pada test
(1.283 teks).

Artefak:

- `results/model_comparison.csv` — tabel perbandingan model tiga kelas.
- `results/confusion_matrix.png` — confusion matrix model biner (final).
- `results/confusion_matrix_3class.png` — confusion matrix model tiga kelas.
- `results/probe_2026.csv` — hasil uji 24 kalimat bergaya 2026.
- `results/error_analysis.csv` — contoh salah prediksi (untuk analisis kualitatif).
- `results/top_features.csv` — kata paling berpengaruh per kelas.
- `results/test_predictions.csv` — seluruh prediksi di test.
- `results/demo_screenshot.png` — tangkapan layar demo web.
- `results/hate_threshold_analysis.csv` — sapuan ambang indikasi hate.
- `results/hate_threshold_curve.png` — kurva precision/recall/F1 terhadap ambang.
- `reports/Laporan_Tugas_Besar_NLP.docx` — laporan siap edit/kumpul.
- `reports/Laporan_Tugas_Besar_NLP.pdf` — versi PDF laporan.
- `reports/presentasi_tugas_besar.pptx` — deck presentasi 14 slide.

## Keputusan metodologis

1. **Dua skema label.** Model tiga kelas (Normal, Abusive, Hate Speech) dilaporkan
   sebagai studi pembanding. Model yang dipakai demo adalah biner
   **Normal vs Kebencian/Abusif**, karena uji generalisasi 2026 menunjukkan batas
   Abusive–Hate Speech tidak dapat dipelajari: 64,7% teks berhinaan di dataset
   memang dilabeli Hate Speech.
2. **Tanpa stemming pada pipeline utama.** Stemming Sastrawi sangat lambat untuk
   13 ribu tweet; char n-gram 3–5 sudah menangani variasi morfologi dan typo.
   Opsi stemming tetap tersedia di `preprocess.py`.
3. **Kamus slang diperbarui.** `data/raw/slang_2026.csv` menambahkan istilah
   seperti *rizz*, *flexing*, *mager*, *ansos*. Kata abusif sengaja tidak
   dinormalisasi karena merupakan sinyal kelas.
4. **Netralisasi nama tokoh politik.** 42 nama dibuang dari teks (training dan
   inference). Tanpa ini, "Jokowi" saja memicu Hate Speech 87,6% karena model
   menghafal polarisasi 2019, bukan mendeteksi ujaran.
5. **Tuning di validation, test sekali.** Model final dipilih dari macro-F1
   validation, bukan dari test, agar evaluasi tidak bias.
6. **Class weight balanced** pada Regresi Logistik dan Linear SVM untuk
   mengatasi kelas minoritas.

## Batasan

- Dataset berasal dari Twitter 2018–2019. Ini sudah diukur: pada 24 kalimat
  bergaya 2026, akurasi skema tiga kelas turun ke 58,3%. Skema biner mencapai
  83,3% pada uji yang sama.
- Anotasi hate speech subjektif dan konteks tweet terpotong dapat mengubah label.
- Model linier mudah diinterpretasi tetapi tidak menangkap konteks panjang.
- Demo ini bukan alat moderasi produksi; keputusan akhir harus melibatkan manusia.

## Uji validitas temporal (opsional)

Isi `data/temporal/temporal_2026.csv` dengan 100–200 komentar terbaru, lalu:

```bash
python src/evaluate_temporal.py
```

Hasilnya di `results/temporal_metrics.json`. Penurunan performa dibanding test
2019 adalah temuan yang dilaporkan sebagai *domain shift*, bukan kegagalan.

## Sitasi

Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive Language
Detection in Indonesian Twitter. *Proceedings of the Third Workshop on Abusive
Language Online*, 46–57. https://doi.org/10.18653/v1/W19-3506

Lisensi dataset: CC BY-NC-SA 4.0 (akademik/non-komersial).
