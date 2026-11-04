"""Build the PDF copies of the documentation.

Uses the same template as the Mathematical Modelling Lab package abstract: A4, Times New
Roman 12pt, a bold centred lab header and document label, a "Team members:" line, bold
section headings, left-aligned 1.5-spaced body text, indented lists and monospaced code
blocks.

Run (reportlab is only needed for this build step):

    uv run --with reportlab python docs/build_pdfs.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "pdf"
LAB = "20XT96 - MATHEMATICAL MODELLING LAB"
TEAM = "22PT17 - Mohamed Aashir S, 21PT10 - Kiruthika V G"
FILES = ["abstract.md", "explanation.md", "demo.md", "architecture.md", "configuration.md"]

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=12, leading=20, alignment=TA_LEFT, spaceAfter=8)
HEADING = ParagraphStyle("heading", fontName="Times-Bold", fontSize=12, leading=20, spaceBefore=10, spaceAfter=4)
LABEL = ParagraphStyle("label", fontName="Times-Bold", fontSize=12, leading=20, alignment=TA_CENTER, spaceAfter=6)
TEAM_HEAD = ParagraphStyle("teamh", fontName="Times-Bold", fontSize=12, leading=20, spaceBefore=10, spaceAfter=2)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=9, leading=11, leftIndent=12, spaceBefore=2, spaceAfter=10)


def inline(text: str) -> str:
    """Convert a small Markdown subset to ReportLab paragraph markup."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def parse(md_text: str):
    """Parse headings, paragraphs, lists and fenced code into (kind, payload) blocks."""
    lines = md_text.splitlines()
    title = None
    blocks: list[tuple[str, object]] = []
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            blocks.append(("para", " ".join(para).strip()))
            para = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if s.startswith("```"):
            flush_para()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
            i += 1
            continue
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip()
            if level == 1 and title is None:
                title = text
            else:
                flush_para()
                blocks.append(("heading", text))
            i += 1
            continue
        if s == "":
            flush_para()
            i += 1
            continue
        if re.match(r"^[-*]\s+", s):
            flush_para()
            items: list[str] = []
            while i < len(lines):
                cur = lines[i].strip()
                m = re.match(r"^[-*]\s+(.*)$", cur)
                if m:
                    items.append(m.group(1))
                elif cur and not cur.startswith(("#", "```")) and not re.match(r"^\d+\.\s+", cur):
                    if not items:
                        break
                    items[-1] += " " + cur
                else:
                    break
                i += 1
            blocks.append(("bullets", items))
            continue
        if re.match(r"^\d+\.\s+", s):
            flush_para()
            items = []
            while i < len(lines):
                cur = lines[i].strip()
                m = re.match(r"^\d+\.\s+(.*)$", cur)
                if m:
                    items.append(m.group(1))
                elif cur and not cur.startswith(("#", "```")) and not re.match(r"^[-*]\s+", cur):
                    if not items:
                        break
                    items[-1] += " " + cur
                else:
                    break
                i += 1
            blocks.append(("numbers", items))
            continue
        para.append(s)
        i += 1

    flush_para()
    return title, blocks


def build(md_path: Path) -> Path:
    title, blocks = parse(md_path.read_text(encoding="utf-8"))
    label = (title or md_path.stem).upper()
    out_path = OUT / f"{md_path.stem}.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2.4 * cm,
        rightMargin=2.4 * cm,
        topMargin=2.4 * cm,
        bottomMargin=2.4 * cm,
        title=title or md_path.stem,
        author="Mohamed Aashir S; Kiruthika V G",
    )

    story = [
        Paragraph(LAB, LABEL),
        Paragraph(label, LABEL),
        Spacer(1, 6),
        Paragraph("Team members:", TEAM_HEAD),
        Paragraph(TEAM, BODY),
    ]
    for kind, payload in blocks:
        if kind == "heading":
            story.append(Paragraph(inline(str(payload)), HEADING))
        elif kind == "para":
            story.append(Paragraph(inline(str(payload)), BODY))
        elif kind == "code":
            story.append(Preformatted(str(payload), CODE))
        elif kind == "bullets":
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(x), BODY), leftIndent=18) for x in payload],
                    bulletType="bullet",
                    start="\u2022",
                    leftIndent=18,
                    bulletIndent=4,
                    bulletFontSize=12,
                    spaceAfter=8,
                )
            )
        elif kind == "numbers":
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(x), BODY), leftIndent=18) for x in payload],
                    bulletType="1",
                    start="1",
                    leftIndent=18,
                    bulletIndent=4,
                    bulletFontSize=12,
                    bulletFormat="%s.",
                    spaceAfter=8,
                )
            )

    doc.build(story)
    return out_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DOCS / name
        if path.exists():
            print(build(path))
    print("PDFs written to", OUT)


if __name__ == "__main__":
    main()
