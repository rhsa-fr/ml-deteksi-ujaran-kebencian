"""Konversi laporan Markdown ke DOCX akademik (Times New Roman, 1,5 spasi).

Konverter ini sengaja dibuat khusus untuk `reports/laporan.md`: mendukung
heading, paragraf, tabel pipe, gambar, blok kode, daftar, dan formula LaTeX
sederhana. Tidak ada dependensi Pandoc.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "laporan.md"
OUTPUT = ROOT / "reports" / "Laporan_Tugas_Besar_NLP.docx"
FORMULA_DIR = ROOT / "results" / "formulas"

FONT = "Times New Roman"
MONO = "Consolas"
BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(0x44, 0x44, 0x44)


# ---------------------------------------------------------------------------
# Helper XML


def shade_cell(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def set_table_borders(table, color: str = "808080", size: str = "4") -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)
        borders.append(element)
    tbl_pr.append(borders)


def remove_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tbl_pr.append(borders)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run.font.name = FONT
    run.font.size = Pt(10)


def add_bottom_border(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


# ---------------------------------------------------------------------------
# Inline formatting


def latex_to_text(value: str) -> str:
    """Ubah LaTeX sederhana menjadi teks yang terbaca di Word."""
    value = value.strip()
    value = value.replace("{,}", ",")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathcal\{X\}", "X", value)
    value = re.sub(r"\\mathcal\{Y\}", "Y", value)
    value = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", value)
    replacements = {
        r"\times": "×",
        r"\mid": "|",
        r"\propto": "∝",
        r"\prod": "∏",
        r"\sum": "Σ",
        r"\log": "log",
        r"\exp": "exp",
        r"\top": "ᵀ",
        r"\xi": "ξ",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\rightarrow": "→",
        r"\in": "∈",
        r"\dots": "…",
        r"\ge": "≥",
        r"\le": "≤",
        r"\cdot": "·",
        r"\qquad": "    ",
        r"\quad": "  ",
        r"\,": " ",
        r"\;": " ",
        r"\{": "{",
        r"\}": "}",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = value.replace("\\", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def add_formula_image(doc: Document, formula: str) -> None:
    """Render formula LaTeX sebagai gambar agar tampil rapi di Word."""
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    normalized = formula.replace("\\[", "").replace("\\]", "")
    normalized = normalized.replace(r"\text{", r"\mathrm{")
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:10]
    path = FORMULA_DIR / f"formula_{digest}.png"
    try:
        figure = plt.figure(figsize=(0.01, 0.01))
        figure.text(0, 0, f"${normalized}$", fontsize=12, color="black")
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            transparent=True,
        )
        plt.close(figure)
        with Image.open(path) as image:
            pixel_width, _ = image.size
        natural_cm = pixel_width / 300 * 2.54
        width_cm = min(max(natural_cm, 5.0), 14.0)
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(6)
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    except Exception:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(latex_to_text(normalized))
        run.italic = True
        run.font.name = FONT
        run.font.size = Pt(11)


def add_inline(paragraph, text: str, size: float = 12, bold: bool = False) -> None:
    """Tambahkan teks dengan **bold**, *italic*, `code`, dan link sederhana."""
    text = re.sub(r"\\\((.*?)\\\)", lambda m: latex_to_text(m.group(1)), text)
    token_pattern = re.compile(
        r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`|\[[^\]]+\]\([^)]+\))"
    )
    for token in token_pattern.split(text):
        if not token:
            continue
        run = paragraph.add_run()
        is_code = False
        if token.startswith("**") and token.endswith("**"):
            run.text = token[2:-2]
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run.text = token[1:-1]
            run.italic = True
        elif token.startswith("`") and token.endswith("`"):
            run.text = token[1:-1]
            is_code = True
            run.font.name = MONO
            run.font.size = Pt(size - 1.5)
        elif token.startswith("[") and "](" in token:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
            run.text = f"{label} ({url})"
        else:
            run.text = token
        if not is_code:
            run.font.name = FONT
        if run.font.size is None:
            run.font.size = Pt(size)
        run.font.color.rgb = BLACK
        if bold:
            run.bold = True


# ---------------------------------------------------------------------------
# Block builders


def add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Heading 1" if level == 1 else "Heading 2"]
    paragraph.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    run.bold = True
    run.font.name = FONT
    run.font.size = Pt(14 if level == 1 else 12)
    run.font.color.rgb = BLACK
    if level == 2:
        run.italic = False


def add_paragraph(doc: Document, text: str, first_indent: bool = True) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Cm(1)
    add_inline(paragraph, text)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text.strip("*"))
    run.italic = True
    run.font.name = FONT
    run.font.size = Pt(10)
    run.font.color.rgb = BLACK


def add_image(doc: Document, path: Path) -> None:
    with Image.open(path) as image:
        width, height = image.size
    picture_width = Cm(11) if height > width else Cm(15)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(path), width=picture_width)


def add_code_block(doc: Document, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="BFBFBF", size="4")
    cell = table.cell(0, 0)
    shade_cell(cell, "F3F3F3")
    first = True
    for line in lines:
        paragraph = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(line if line else " ")
        run.font.name = MONO
        run.font.size = Pt(8.5)
        run.font.color.rgb = BLACK
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.line_spacing = 1.0


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    # Ulangi header jika tabel terpotong halaman, dan jangan pecah baris.
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    header_flag = OxmlElement("w:tblHeader")
    header_flag.set(qn("w:val"), "true")
    header_pr.append(header_flag)
    for row in table.rows:
        row_pr = row._tr.get_or_add_trPr()
        row_pr.append(OxmlElement("w:cantSplit"))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if col_index >= len(rows[0]):
                continue
            cell = table.cell(row_index, col_index)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.0
            add_inline(paragraph, value.replace("<br>", "\n"), size=9)
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True
                shade_cell(cell, "E7E6E6")
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)
    spacer.paragraph_format.line_spacing = 1.0


def add_cover_table(doc: Document, rows: list[list[str]]) -> None:
    if rows and all(not cell.strip() for cell in rows[0]):
        rows = rows[1:]
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    remove_table_borders(table)
    for row_index, row in enumerate(rows):
        for col_index in range(2):
            cell = table.cell(row_index, col_index)
            cell.width = Cm(4.2 if col_index == 0 else 9.8)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.15
            text = row[col_index] if col_index < len(row) else ""
            run = paragraph.add_run(text)
            run.font.name = FONT
            run.font.size = Pt(11)
            run.font.color.rgb = BLACK
            if col_index == 0:
                run.bold = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(6)


# ---------------------------------------------------------------------------
# Parser


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        raw = lines[index].strip().strip("|")
        cells = [cell.strip() for cell in raw.split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()

    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21)
    section.top_margin = Cm(4)
    section.bottom_margin = Cm(3)
    section.left_margin = Cm(3)
    section.right_margin = Cm(3)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)

    # ------------------------- cover
    title = ""
    subtitle = ""
    cover_rows: list[list[str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("**") and line.endswith("**") and not subtitle:
            subtitle = line.strip("*")
        elif line.startswith("|"):
            cover_rows, index = parse_table(lines, index)
            break
        index += 1

    add_paragraph(doc, "LAPORAN TUGAS BESAR", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].runs[0].bold = True
    doc.paragraphs[-1].runs[0].font.size = Pt(12)
    doc.paragraphs[-1].paragraph_format.space_before = Pt(24)

    add_paragraph(doc, title, first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.line_spacing = 1.15
    doc.paragraphs[-1].paragraph_format.space_before = Pt(18)
    for run in doc.paragraphs[-1].runs:
        run.bold = True
        run.font.size = Pt(16)

    if subtitle:
        add_paragraph(doc, subtitle, first_indent=False)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_before = Pt(8)
        for run in doc.paragraphs[-1].runs:
            run.bold = True
            run.font.size = Pt(12)

    line_paragraph = doc.add_paragraph()
    add_bottom_border(line_paragraph)
    line_paragraph.paragraph_format.space_after = Pt(18)

    add_paragraph(doc, "Disusun oleh:", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].runs[0].bold = True
    doc.paragraphs[-1].paragraph_format.space_after = Pt(8)
    add_cover_table(doc, cover_rows)

    add_paragraph(doc, "PROGRAM STUDI D4 REKAYASA PERANGKAT LUNAK", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].runs[0].bold = True
    add_paragraph(doc, "JURUSAN TEKNIK INFORMATIKA", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(doc, "POLITEKNIK NEGERI INDRAMAYU", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(doc, "NOVEMBER 2026", first_indent=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # ------------------------- body
    # Mulai dari Abstrak; lewati baris cover.
    body_start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("## Abstrak")),
        0,
    )
    index = body_start
    in_code = False
    code_lines: list[str] = []
    in_references = False

    while index < len(lines):
        raw = lines[index]
        line = raw.strip()

        if in_code:
            if line.startswith("```"):
                add_code_block(doc, code_lines)
                code_lines = []
                in_code = False
            else:
                code_lines.append(raw)
            index += 1
            continue

        if line.startswith("```"):
            in_code = True
            index += 1
            continue

        if not line:
            index += 1
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            if heading.startswith(("1. ", "Daftar Pustaka", "Lampiran")):
                doc.add_page_break()
            in_references = heading.startswith("Daftar Pustaka")
            add_heading(doc, heading, 1)
            index += 1
            continue

        if line.startswith("### "):
            add_heading(doc, line[4:].strip(), 2)
            index += 1
            continue

        if line.startswith("!["):
            match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if match:
                image_path = (SOURCE.parent / match.group(1)).resolve()
                if image_path.exists():
                    add_image(doc, image_path)
            index += 1
            continue

        if line.startswith("|"):
            rows, index = parse_table(lines, index)
            add_table(doc, rows)
            continue

        if line == "---":
            paragraph = doc.add_paragraph()
            add_bottom_border(paragraph)
            paragraph.paragraph_format.space_after = Pt(8)
            index += 1
            continue

        if line.startswith("> "):
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.8)
            paragraph.paragraph_format.line_spacing = 1.3
            run = paragraph.add_run(line[2:].strip())
            run.italic = True
            run.font.name = FONT
            run.font.size = Pt(11)
            run.font.color.rgb = GRAY
            index += 1
            continue

        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.line_spacing = 1.3
            paragraph.paragraph_format.space_after = Pt(2)
            add_inline(paragraph, line[2:].strip())
            index += 1
            continue

        if re.match(r"^\d+\.\s", line):
            # Nomor manual agar penomoran tidak berlanjut antar-bab.
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(1.0)
            paragraph.paragraph_format.first_line_indent = Cm(-0.75)
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.space_after = Pt(0)
            add_inline(paragraph, line)
            index += 1
            continue

        if line.startswith("\\["):
            formula_lines = [line]
            while not formula_lines[-1].strip().endswith("\\]") and index + 1 < len(lines):
                index += 1
                formula_lines.append(lines[index].strip())
            formula = " ".join(formula_lines).replace("\\[", "").replace("\\]", "")
            add_formula_image(doc, formula)
            index += 1
            continue

        # caption italic: *Gambar ...*
        if line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            add_caption(doc, line)
            index += 1
            continue

        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.line_spacing = 1.5
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if not line.startswith(("**Kata kunci", "**Kata Kunci")):
            paragraph.paragraph_format.first_line_indent = Cm(1)
        if in_references:
            paragraph.paragraph_format.left_indent = Cm(1)
            paragraph.paragraph_format.first_line_indent = Cm(-1)
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_inline(paragraph, line)
        index += 1

    # Penutup jika code block tidak ditutup
    if code_lines:
        add_code_block(doc, code_lines)

    doc.save(OUTPUT)
    print(f"DOCX tersimpan di {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
