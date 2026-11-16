"""Bangun deck presentasi PPTX bergaya brutalist dari hasil training."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "presentasi_tugas_besar.pptx"

PAPER = RGBColor(0xF2, 0xEF, 0xE4)
INK = RGBColor(0x0A, 0x0A, 0x0A)
WHITE = RGBColor(0xFF, 0xFD, 0xF5)
LIME = RGBColor(0xC8, 0xFF, 0x00)
ORANGE = RGBColor(0xFF, 0xB0, 0x00)
PINK = RGBColor(0xFF, 0x4F, 0xA3)
BLUE = RGBColor(0x2F, 0x5B, 0xFF)
RED = RGBColor(0xFF, 0x3B, 0x30)
MUTED = RGBColor(0x5B, 0x5B, 0x52)

DISPLAY = "Arial Black"
MONO = "Consolas"
BODY = "Arial"

METRICS = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
BEST = METRICS["best_metrics"]
COMPARISON = {
    "nb": METRICS["models"]["nb"]["test"],
    "lr": METRICS["models"]["lr"]["test"],
    "svm": METRICS["models"]["svm"]["test"],
}

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = PAPER


def add_rect(slide, x, y, w, h, color, line=INK, line_width=3):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = line
    shape.line.width = Pt(line_width)
    shape.shadow.inherit = False
    return shape


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=14,
    color=INK,
    bold=False,
    font=BODY,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    line_spacing=1.05,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    lines = text.split("\n")
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = line_spacing
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = font
        run.font.color.rgb = color
    return box


def ticker(slide, text: str) -> None:
    add_rect(slide, 0, 0, 13.333, 0.30, INK, INK, 0)
    add_text(slide, 0.45, 0.055, 12.4, 0.2, text, 8.5, LIME, font=MONO)


def header(slide, number: str, kicker: str, title: str) -> None:
    ticker(slide, "NLP/26  ///  TUGAS BESAR KECERDASAN BUATAN 2026  ///  DETEKSI UJARAN KEBENCIAN")
    add_text(slide, 0.6, 0.55, 12, 0.25, f"[{number}] {kicker}", 11, MUTED, True, MONO)
    add_text(slide, 0.6, 0.86, 12.1, 0.8, title, 27, INK, True, DISPLAY)


def footer(slide, number: str) -> None:
    add_rect(slide, 0, 7.28, 13.333, 0.22, INK, INK, 0)
    add_text(slide, 0.45, 7.305, 10, 0.14, "DEMO AKADEMIK · BUKAN ALAT MODERASI PRODUKSI", 7.5, LIME, font=MONO)
    add_text(slide, 12.2, 7.305, 0.8, 0.14, number, 7.5, LIME, font=MONO, align=PP_ALIGN.RIGHT)


def new_slide(number: str, kicker: str, title: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    header(slide, number, kicker, title)
    return slide


# ---------------------------------------------------------------- slide 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
ticker(slide, "13.169 TWEET  ///  TF-IDF  ///  NAIVE BAYES · LOGREG · SVM  ///  MACRO-F1 0,8075")
add_rect(slide, 0.6, 0.85, 1.55, 0.62, INK, INK, 0)
add_text(slide, 0.72, 1.00, 1.4, 0.35, "NLP/26", 18, LIME, True, DISPLAY)
add_text(slide, 2.35, 0.95, 7, 0.5, "LABORATORIUM KECERDASAN BUATAN · TUGAS BESAR", 10, MUTED, True, MONO)
add_text(slide, 0.6, 1.85, 12.2, 0.4, "[01] KLASIFIKASI TEKS BAHASA INDONESIA", 12, INK, True, MONO)
add_text(slide, 0.6, 2.25, 12.2, 1.0, "DETEKSI", 62, INK, True, DISPLAY)
add_rect(slide, 0.6, 3.28, 5.9, 1.05, LIME, INK, 4)
add_text(slide, 0.78, 3.43, 5.7, 0.8, "UJARAN", 58, INK, True, DISPLAY)
add_text(slide, 0.6, 4.45, 12.2, 1.0, "KEBENCIAN.", 62, INK, True, DISPLAY)
add_text(
    slide,
    0.6,
    5.55,
    8.4,
    0.9,
    "Tiga kelas: Normal, Abusive, Hate Speech.\nPipeline: praproses → TF-IDF → tiga model klasik → evaluasi → demo web.",
    13,
    INK,
    False,
    BODY,
)
add_rect(slide, 9.5, 5.25, 3.2, 1.35, PINK, INK, 3)
add_text(slide, 9.68, 5.42, 2.9, 0.9, "BENCHMARK 2019\nDIUJI ULANG 2026", 13, INK, True, MONO)
add_text(slide, 0.6, 6.75, 12, 0.3, "Kelompok [isi] · [Nama] / [NIM] · November 2026", 10, MUTED, True, MONO)
footer(slide, "01")

# ---------------------------------------------------------------- slide 2
slide = new_slide("02", "MASALAH", "Moderasi manual tidak skalabel.")
cards = [
    ("13.169", "tweet pada dataset\n2018–2019", LIME),
    ("3", "kelas yang harus\ndibedakan model", ORANGE),
    ("0,8075", "macro-F1 model\nterbaik di test", PINK),
]
for i, (big, small, color) in enumerate(cards):
    x = 0.6 + i * 4.1
    add_rect(slide, x, 2.0, 3.75, 2.0, color, INK, 3)
    add_text(slide, x + 0.25, 2.2, 3.2, 0.7, big, 42, INK, True, DISPLAY)
    add_text(slide, x + 0.25, 3.0, 3.2, 0.8, small, 12, INK, True, MONO)
add_text(
    slide,
    0.6,
    4.45,
    12.2,
    1.6,
    "Ujaran kebencian menyebar lebih cepat daripada kemampuan tim moderasi membaca.\n"
    "Pendekatan daftar kata gagal pada sarkasme, negasi, dan variasi ejaan.\n"
    "Model perlu menjadi penyaring awal, bukan hakim akhir.",
    19,
    INK,
    False,
    BODY,
    line_spacing=1.35,
)
add_rect(slide, 0.6, 6.15, 12.1, 0.75, WHITE, INK, 3)
add_text(slide, 0.8, 6.3, 11.7, 0.45, "PERTANYAAN: model klasik sederhana seberapa jauh bisa membantu?", 14, INK, True, MONO)
footer(slide, "02")

# ---------------------------------------------------------------- slide 3
slide = new_slide("03", "DATASET", "13.169 tweet, 12 label asli.")
add_text(slide, 0.6, 1.85, 5.7, 0.4, "SUMBER", 11, MUTED, True, MONO)
add_text(slide, 0.6, 2.15, 5.6, 1.2, "Ibrohim & Budi (2019)\nMulti-label Hate Speech and Abusive\nLanguage Detection in Indonesian Twitter", 15, INK, True, BODY)
add_text(slide, 0.6, 3.55, 5.6, 0.4, "SETELAH DEDUP", 11, MUTED, True, MONO)
add_text(slide, 0.6, 3.85, 5.6, 0.6, "12.828 teks unik  (341 duplikat dibuang)", 15, INK, True, BODY)
add_text(slide, 0.6, 4.6, 5.6, 0.4, "PEMETAAN 3 KELAS", 11, MUTED, True, MONO)
add_text(slide, 0.6, 4.9, 5.6, 1.3, "HS=1 → Hate Speech\nAbusive=1 tanpa HS → Abusive\nsisanya → Normal", 14, INK, False, MONO, line_spacing=1.25)

data = [("Normal", 5701, LIME), ("Abusive", 1709, ORANGE), ("Hate Speech", 5418, RED)]
max_value = max(value for _, value, _ in data)
for i, (label, value, color) in enumerate(data):
    y = 2.1 + i * 1.35
    add_text(slide, 6.6, y, 1.4, 0.3, label, 11, INK, True, MONO)
    add_rect(slide, 8.0, y - 0.02, 4.0, 0.42, WHITE, INK, 2)
    add_rect(slide, 8.0, y - 0.02, 4.0 * value / max_value, 0.42, color, INK, 2)
    add_text(slide, 12.15, y, 0.7, 0.3, f"{value:,}".replace(",", "."), 10, INK, True, MONO, PP_ALIGN.RIGHT)
add_text(slide, 6.6, 6.2, 6.2, 0.6, "Distribusi tidak seimbang: Abusive hanya 13,3%.", 12, INK, True, MONO)
footer(slide, "03")

# ---------------------------------------------------------------- slide 4
slide = new_slide("04", "PIPELINE", "Tujuh tahap yang dapat direproduksi.")
steps = [
    ("01", "DATASET", "13.169 tweet mentah", LIME),
    ("02", "PRAPROSES", "case folding · slang 2026 · dedup", WHITE),
    ("03", "SPLIT", "10.262 / 1.283 / 1.283", ORANGE),
    ("04", "TF-IDF", "word 1–2 + char_wb 3–5", WHITE),
    ("05", "MODEL", "NB · LogReg · Linear SVM", BLUE),
    ("06", "EVALUASI", "macro-F1 · confusion matrix", WHITE),
    ("07", "DEMO", "FastAPI + HTML", PINK),
]
for i, (num, name, detail, color) in enumerate(steps):
    y = 1.75 + i * 0.72
    add_rect(slide, 0.6, y, 7.4, 0.6, color, INK, 2)
    add_text(slide, 0.75, y + 0.12, 0.5, 0.3, num, 11, INK, True, MONO)
    add_text(slide, 1.25, y + 0.12, 1.6, 0.3, name, 12, INK, True, DISPLAY)
    add_text(slide, 2.95, y + 0.15, 4.8, 0.3, detail, 9.5, INK, False, MONO)
add_text(slide, 8.5, 1.9, 4.2, 0.4, "PRINSIP", 11, MUTED, True, MONO)
add_text(slide, 8.5, 2.3, 4.3, 3.6, "Setiap tahap wajib punya:\n\nKONSEP\nteori yang mendasari\n\nIMPLEMENTASI\nkode yang dijalankan\n\nHASIL\nangka dan artefak\n\nANALISIS\nkenapa hasilnya begitu", 12, INK, False, MONO, line_spacing=1.05)
add_rect(slide, 8.5, 6.35, 4.2, 0.65, LIME, INK, 3)
add_text(slide, 8.7, 6.46, 3.9, 0.45, "Reproducible dari unduh data sampai demo.", 10, INK, True, MONO)
footer(slide, "04")

# ---------------------------------------------------------------- slide 5
slide = new_slide("05", "PRAPROSES", "Bersihkan noise, jangan hapus sinyal.")
add_rect(slide, 0.6, 1.85, 5.6, 0.45, INK, INK, 0)
add_text(slide, 0.75, 1.95, 5.3, 0.3, "SEBELUM", 11, LIME, True, MONO)
add_text(slide, 0.6, 2.4, 5.7, 1.7, "@USER goblokkk bgt sih loe, gak\nusah sok tau ah\n#2019GantiPresiden", 14, INK, False, MONO, line_spacing=1.3)
add_rect(slide, 6.6, 1.85, 6.1, 0.45, LIME, INK, 3)
add_text(slide, 6.75, 1.95, 5.8, 0.3, "SESUDAH", 11, INK, True, MONO)
add_text(slide, 6.6, 2.4, 6.1, 1.7, "goblok banget sih kamu tidak\nusaha sok tahu ah\nganti presiden", 14, INK, False, MONO, line_spacing=1.3)
steps = [
    ("case folding", "huruf kecil"),
    ("URL/mention", "dibuang"),
    ("karakter berulang", "goblokkk → goblok"),
    ("slang 2026", "rizz → pesona"),
    ("kata abusif", "TIDAK dinormalisasi"),
    ("stopword", "TIDAK dihapus"),
]
for i, (title, detail) in enumerate(steps):
    x = 0.6 + (i % 3) * 4.15
    y = 4.5 + (i // 3) * 1.0
    add_rect(slide, x, y, 3.85, 0.8, WHITE, INK, 2)
    add_text(slide, x + 0.15, y + 0.1, 3.5, 0.25, title.upper(), 10, INK, True, MONO)
    add_text(slide, x + 0.15, y + 0.38, 3.5, 0.3, detail, 10, MUTED, False, MONO)
footer(slide, "05")

# ---------------------------------------------------------------- slide 6
slide = new_slide("06", "FITUR", "TF-IDF: kata dan karakter.")
add_rect(slide, 0.6, 1.9, 5.9, 2.1, LIME, INK, 3)
add_text(slide, 0.8, 2.05, 5.5, 0.3, "WORD 1–2 GRAM", 12, INK, True, MONO)
add_text(slide, 0.8, 2.4, 5.5, 1.4, "15.315 fitur\nmenangkap kata dan frasa:\n“tidak pantas”, “ganti presiden”", 13, INK, False, MONO, line_spacing=1.25)
add_rect(slide, 6.8, 1.9, 5.9, 2.1, ORANGE, INK, 3)
add_text(slide, 7.0, 2.05, 5.5, 0.3, "CHAR_WB 3–5 GRAM", 12, INK, True, MONO)
add_text(slide, 7.0, 2.4, 5.5, 1.4, "50.000 fitur\nmenangkap typo dan variasi:\ngoblok, goblokk, goblokkk", 13, INK, False, MONO, line_spacing=1.25)
add_rect(slide, 0.6, 4.35, 12.1, 1.35, WHITE, INK, 3)
add_text(slide, 0.85, 4.55, 11.6, 0.9, "tf-idf(t,d) = tf(t,d) × log((1+N)/(1+df(t))) + 1\nmin_df=3 · max_df=0,95 · sublinear_tf=True · total 65.315 fitur", 14, INK, False, MONO, line_spacing=1.3)
add_text(slide, 0.6, 6.0, 12.1, 0.7, "Dimensi tinggi bukan masalah untuk model linier: yang dihitung hanya fitur non-nol.", 12, MUTED, True, MONO)
footer(slide, "06")

# ---------------------------------------------------------------- slide 7
slide = new_slide("07", "MODEL", "Tiga cara berpikir, satu pipeline.")
models = [
    ("NAIVE BAYES", "Teorema Bayes +\nindependensi fitur.\nα = 0,1", LIME),
    ("REGRESI LOGISTIK", "Cross-entropy +\nsoftmax multikelas.\nC = 1,0", ORANGE),
    ("LINEAR SVM", "Margin maksimum +\nhinge loss.\nC = 0,1", BLUE),
]
for i, (name, detail, color) in enumerate(models):
    x = 0.6 + i * 4.1
    add_rect(slide, x, 1.95, 3.75, 2.5, color, INK, 3)
    text_color = WHITE if color == BLUE else INK
    add_text(slide, x + 0.2, 2.1, 3.35, 0.35, name, 13, text_color, True, DISPLAY)
    add_text(slide, x + 0.2, 2.6, 3.35, 1.5, detail, 12, text_color, False, MONO, line_spacing=1.3)
add_text(slide, 0.6, 4.85, 12.1, 0.4, "YANG DIBANDINGKAN", 11, MUTED, True, MONO)
add_text(slide, 0.6, 5.2, 12.1, 1.3, "Konfigurasi terbaik dipilih di validation set.\nTest set hanya dipakai sekali untuk evaluasi akhir.\n5-fold CV pada train untuk melihat stabilitas.", 15, INK, False, BODY, line_spacing=1.3)
add_rect(slide, 0.6, 6.45, 5.5, 0.55, LIME, INK, 3)
add_text(slide, 0.75, 6.57, 5.2, 0.3, "BUKAN cherry-picking di test.", 11, INK, True, MONO)
footer(slide, "07")

# ---------------------------------------------------------------- slide 8
slide = new_slide("08", "HASIL", "Linear SVM menang tipis.")
rows = [
    ("Model", "Val F1", "Acc", "Macro-F1", "Weighted-F1"),
    ("Naive Bayes", "0,7698", "79,89%", "0,7601", "0,8018"),
    ("Regresi Logistik", "0,8023", "82,00%", "0,7859", "0,8216"),
    ("Linear SVM", "0,8056", "83,71%", "0,8075", "0,8365"),
]
table_shape = slide.shapes.add_table(4, 5, Inches(0.6), Inches(1.95), Inches(12.1), Inches(2.8))
table = table_shape.table
table.columns[0].width = Inches(3.6)
table.columns[1].width = Inches(2.0)
table.columns[2].width = Inches(2.0)
table.columns[3].width = Inches(2.2)
table.columns[4].width = Inches(2.3)
for r, row in enumerate(rows):
    for c, value in enumerate(row):
        cell = table.cell(r, c)
        cell.text = value
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.fill.solid()
        if r == 0:
            cell.fill.fore_color.rgb = INK
            color = LIME
        elif r == 3:
            cell.fill.fore_color.rgb = LIME
            color = INK
        else:
            cell.fill.fore_color.rgb = WHITE
            color = INK
        for paragraph in cell.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(11)
                run.font.name = MONO
                run.font.bold = r in (0, 3)
                run.font.color.rgb = color
add_text(slide, 0.6, 5.1, 12.1, 0.4, "KENAPA SVM?", 11, MUTED, True, MONO)
add_text(slide, 0.6, 5.45, 12.1, 1.1, "Data teks berdimensi tinggi cenderung linear-separable.\nMargin maksimum memberi regularisasi alami; LR sedikit kalah, NB paling cepat tetapi asumsinya paling kuat.", 13, INK, False, BODY, line_spacing=1.3)
add_text(slide, 0.6, 6.7, 12.1, 0.3, "Sumber: results/model_comparison.csv · results/metrics.json", 9, MUTED, False, MONO)
footer(slide, "08")

# ---------------------------------------------------------------- slide 9
slide = new_slide("09", "EVALUASI", "Confusion matrix: di mana model salah?")
slide.shapes.add_picture(str(RESULTS / "confusion_matrix.png"), Inches(0.6), Inches(1.85), height=Inches(4.9))
add_text(slide, 7.0, 1.9, 5.7, 0.35, "DARI 1.283 TEKS UJI", 11, MUTED, True, MONO)
add_text(slide, 7.0, 2.25, 5.7, 0.9, "1.074 benar\n209 salah", 30, INK, True, DISPLAY, line_spacing=1.0)
add_text(slide, 7.0, 3.35, 5.7, 2.8, "Hate Speech → Normal: 70\nNormal → Hate Speech: 42\nAbusive → Hate Speech: 35\nHate Speech → Abusive: 26\n\nKesalahan terbesar: batas\nHate Speech ↔ Normal (112).", 12.5, INK, False, MONO, line_spacing=1.25)
add_rect(slide, 7.0, 6.2, 5.5, 0.55, ORANGE, INK, 3)
add_text(slide, 7.15, 6.32, 5.2, 0.3, "Abusive adalah kelas tersulit.", 11, INK, True, MONO)
footer(slide, "09")

# ---------------------------------------------------------------- slide 10
slide = new_slide("10", "PER KELAS", "Abusive masih tertinggal.")
per_class = [("Normal", 0.8729, LIME), ("Abusive", 0.7122, ORANGE), ("Hate Speech", 0.8376, RED)]
for i, (label, value, color) in enumerate(per_class):
    y = 1.95 + i * 1.45
    add_text(slide, 0.6, y, 1.8, 0.35, label, 12, INK, True, MONO)
    add_rect(slide, 2.5, y, 7.0, 0.55, WHITE, INK, 2)
    add_rect(slide, 2.5, y, 7.0 * value, 0.55, color, INK, 2)
    add_text(slide, 9.7, y + 0.05, 1.2, 0.4, f"{value:.4f}".replace(".", ","), 14, INK, True, MONO)
add_text(slide, 0.6, 6.35, 12.2, 0.5, "Macro-F1 0,8075 — rata-rata F1 tiga kelas, bukan akurasi keseluruhan.", 11, MUTED, True, MONO)
footer(slide, "10")

# ---------------------------------------------------------------- slide 11
slide = new_slide("11", "ERROR ANALYSIS", "Kesalahan yang paling mahal.")
errors = [
    ("HATE → NORMAL", "70 teks", "Sarkasme tanpa kata kasar; model kehilangan sinyal leksikal.", ORANGE),
    ("ABUSIVE → HATE", "35 teks", "Kata kasar + kata kelompok; batas kelas kabur.", PINK),
    ("NORMAL → HATE", "42 teks", "Topik politik/agama memicu false positive.", LIME),
]
for i, (title, count, detail, color) in enumerate(errors):
    x = 0.6 + i * 4.1
    add_rect(slide, x, 1.95, 3.75, 2.9, color, INK, 3)
    add_text(slide, x + 0.2, 2.1, 3.35, 0.35, title, 11, INK, True, MONO)
    add_text(slide, x + 0.2, 2.5, 3.35, 0.6, count, 26, INK, True, DISPLAY)
    add_text(slide, x + 0.2, 3.3, 3.35, 1.3, detail, 11, INK, False, BODY, line_spacing=1.25)
add_text(slide, 0.6, 5.2, 12.2, 1.6, "Model bag-of-words tidak memahami negasi jangka panjang, pelaku, target, dan konteks percakapan.\nContoh di laporan disensor; data mentah tetap tersedia untuk audit internal.", 13, INK, False, BODY, line_spacing=1.3)
footer(slide, "11")

# ---------------------------------------------------------------- slide 12
slide = new_slide("12", "DEMO", "Model yang bisa dicoba.")
slide.shapes.add_picture(str(RESULTS / "demo_screenshot.png"), Inches(0.6), Inches(1.8), height=Inches(5.1))
add_text(slide, 6.8, 1.9, 5.9, 0.35, "FITUR DEMO", 11, MUTED, True, MONO)
add_text(slide, 6.8, 2.3, 5.9, 3.9, "Input teks bebas\nLabel + confidence\nSkor tiga kelas\nToken pendorong / penahan\nTeks setelah praproses\n\nEndpoint: /api/predict,\n/api/model-info, /api/health,\n/docs", 13, INK, False, MONO, line_spacing=1.3)
add_rect(slide, 6.8, 6.3, 5.7, 0.55, LIME, INK, 3)
add_text(slide, 6.95, 6.42, 5.4, 0.3, "buka di http://localhost:8000", 10, INK, True, MONO)
footer(slide, "12")

# ---------------------------------------------------------------- slide 13
slide = new_slide("13", "BATASAN", "Jangan klaim lebih dari yang diuji.")
limits = [
    ("DATASET 2019", "Twitter 2018–2019; bahasa dan platform sudah berubah."),
    ("DOMAIN SHIFT", "Belum diuji pada data 2026; template uji temporal tersedia."),
    ("BIAS POLITIK", "Token seperti cebong/ahok/prabowo menjadi pintasan."),
    ("BUKAN PRODUKSI", "Perlu tinjauan manusia; skor SVM belum terkalibrasi."),
]
for i, (title, detail) in enumerate(limits):
    x = 0.6 + (i % 2) * 6.15
    y = 1.95 + (i // 2) * 2.15
    add_rect(slide, x, y, 5.8, 1.75, WHITE, INK, 3)
    add_text(slide, x + 0.2, y + 0.15, 5.4, 0.3, title, 11, INK, True, MONO)
    add_text(slide, x + 0.2, y + 0.55, 5.4, 1.0, detail, 12, INK, False, BODY, line_spacing=1.25)
footer(slide, "13")

# ---------------------------------------------------------------- slide 14
slide = new_slide("14", "KESIMPULAN", "Baseline kuat, langkah berikutnya jelas.")
add_rect(slide, 0.6, 1.95, 3.9, 1.7, LIME, INK, 3)
add_text(slide, 0.8, 2.1, 3.5, 0.5, "0,8075", 30, INK, True, DISPLAY)
add_text(slide, 0.8, 2.7, 3.5, 0.5, "macro-F1 SVM", 11, INK, True, MONO)
add_rect(slide, 4.7, 1.95, 3.9, 1.7, ORANGE, INK, 3)
add_text(slide, 4.9, 2.1, 3.5, 0.5, "83,71%", 30, INK, True, DISPLAY)
add_text(slide, 4.9, 2.7, 3.5, 0.5, "akurasi test", 11, INK, True, MONO)
add_rect(slide, 8.8, 1.95, 3.9, 1.7, PINK, INK, 3)
add_text(slide, 9.0, 2.1, 3.5, 0.5, "1.283", 30, INK, True, DISPLAY)
add_text(slide, 9.0, 2.7, 3.5, 0.5, "teks uji", 11, INK, True, MONO)
add_text(slide, 0.6, 4.0, 12.2, 0.35, "NEXT", 11, MUTED, True, MONO)
add_text(slide, 0.6, 4.35, 12.2, 2.3, "1. Fine-tune IndoBERT sebagai pembanding.\n2. Uji temporal 2026 dengan 100–200 sampel baru.\n3. Kalibrasi probabilitas dan analisis bias.\n4. Tambahkan fitur konteks/negasi sebelum produksi.", 15, INK, False, BODY, line_spacing=1.35)
footer(slide, "14")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"Deck tersimpan di {OUT}")
