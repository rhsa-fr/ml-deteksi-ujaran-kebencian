# Berkas Laporan dan Presentasi

| Berkas | Isi |
|---|---|
| `laporan.md` | Sumber laporan lengkap (Markdown) |
| `Laporan_Tugas_Besar_NLP.docx` | Laporan siap edit/kumpul (Times New Roman, 1,5 spasi, margin 4-3-3-3) |
| `Laporan_Tugas_Besar_NLP.pdf` | Versi PDF untuk pratinjau |
| `presentasi_tugas_besar.pptx` | Deck presentasi 14 slide, tema brutalist |
| `slides.md` | Outline slide + catatan pembicara |
| `slides_png/` | Hasil ekspor slide (artefak build, tidak perlu dikumpulkan) |
| `pdf_pages/` | Hasil render PDF (artefak build, tidak perlu dikumpulkan) |

## Yang masih harus diisi manual

1. **Identitas kelompok** di cover `laporan.md`: nomor kelompok, nama, NIM.
   Setelah diubah, jalankan ulang:

   ```bash
   python src/build_docx.py
   ```

2. **Nama/NIM di slide 1** bisa diedit langsung di PowerPoint, atau ubah
   `src/build_slides.py` lalu jalankan `python src/build_slides.py`.

3. **Screenshot demo** di `results/demo_screenshot.png` sudah tersedia. Jika
   tampilan diubah, ambil ulang dengan Edge headless:

   ```powershell
   & "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
     --headless=new --disable-gpu --hide-scrollbars --window-size=1440,1500 `
     --virtual-time-budget=9000 --screenshot="results/demo_screenshot.png" `
     "http://127.0.0.1:8000/?demo=hate"
   ```

## Membangun ulang

```bash
python src/build_docx.py       # DOCX dari laporan.md
python src/build_slides.py     # PPTX dari metrics.json
```

Konversi DOCX ke PDF memakai Microsoft Word (opsional):

```powershell
$word = New-Object -ComObject Word.Application
$doc = $word.Documents.Open("D:\Project\AI\tugas-besar-nlp\reports\Laporan_Tugas_Besar_NLP.docx")
$doc.SaveAs("D:\Project\AI\tugas-besar-nlp\reports\Laporan_Tugas_Besar_NLP.pdf", 17)
$doc.Close(); $word.Quit()
```

## Catatan

- Gambar dan tabel di laporan memakai angka asli dari `results/metrics.json`.
- Formula dirender sebagai gambar oleh `build_docx.py` agar tidak menjadi
  teks LaTeX mentah di Word.
- Jangan mengubah angka di laporan tanpa menjalankan ulang `src/train.py`.
