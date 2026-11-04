# GenomeFolio Documentation

Documentation for the GenomeFolio package (20XT87 - Data Mining Lab).

**Authors:** Mohamed Aashir S (22PT17) and Kiruthika V G (21PT10)

## Contents

- [Package Abstract](abstract.md) — one-page overview of the problem, method and outcome. PDF: `pdf/abstract.pdf`
- [Project Explanation](explanation.md) — the full "why" and "how": symbolization, alignment, clustering and forecasting. PDF: `pdf/explanation.pdf`
- [Demo Guide](demo.md) — install, run, sample output and what to try. PDF: `pdf/demo.pdf`
- [System Architecture](architecture.md) — modules, data flow, config schema and tests. PDF: `pdf/architecture.pdf`
- [Configuration Reference](configuration.md) — every parameter and its meaning. PDF: `pdf/configuration.pdf`

## Building the PDFs

The PDF versions reuse the Data Mining Lab abstract template (A4, Times New Roman 12pt,
centred lab header, `Team members:` line, bold section headings). Each Markdown file is
rendered to `docs/pdf/<name>.pdf` with:

```
uv run --with reportlab python docs/build_pdfs.py
```

The original lab abstract is kept for reference at `pdf/Abstract_original_template.pdf`.
