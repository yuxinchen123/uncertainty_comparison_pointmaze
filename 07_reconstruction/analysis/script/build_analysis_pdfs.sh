#!/usr/bin/env bash
# Build pdf1 (KaTeX HTML → headless Chromium) and pdf2 (pdflatex) from analysis.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${ROOT}/pdf"
MD="${ROOT}/analysis.md"
PANDOC="${ROOT}/.tools/pandoc"
if [[ ! -x "${PANDOC}" ]]; then
  PANDOC="$(command -v pandoc || true)"
fi
if [[ -z "${PANDOC}" ]]; then
  echo "Install pandoc or unpack https://github.com/jgm/pandoc/releases to ${ROOT}/.tools/" >&2
  exit 1
fi

TMP_HTML="$(mktemp "${TMPDIR:-/tmp}/analysis_katex.XXXXXX.html)"
TMP_MD="$(mktemp "${TMPDIR:-/tmp}/analysis_pdflatex.XXXXXX.md)"
cleanup() { rm -f "${TMP_HTML}" "${TMP_MD}"; }
trap cleanup EXIT

"${PANDOC}" "${MD}" -s -o "${TMP_HTML}" --katex --resource-path="${ROOT}:${ROOT}/.."

python3 << PY
from pathlib import Path
from playwright.sync_api import sync_playwright

html_path = Path("${TMP_HTML}").resolve()
pdf_path = Path("${PDF_DIR}/pdf1.pdf")
pdf_path.parent.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(html_path.as_uri(), wait_until="networkidle", timeout=120_000)
    page.wait_for_timeout(2000)
    page.pdf(
        path=str(pdf_path),
        format="A4",
        print_background=True,
        margin={"top": "12mm", "bottom": "12mm", "left": "14mm", "right": "14mm"},
    )
    browser.close()
print("Wrote", pdf_path)
PY

python3 << PY
from pathlib import Path
src = Path("${MD}")
dst = Path("${TMP_MD}")
text = src.read_text(encoding="utf-8")
for a, b in [
    ("β", r"$\beta$"),
    ("α", r"$\alpha$"),
    ("ρ", r"$\rho$"),
    ("–", "--"),
    ("—", "---"),
    ("\u2019", "'"),
    ("\u201c", "``"),
    ("\u201d", "''"),
    ("…", r"\\ldots"),
    ("→", r"$\rightarrow$"),
    ("·", r"$\\cdot$"),
    ("≪", r"$\ll$"),
    ("≫", r"$\gg$"),
    ("\u202f", " "),
]:
    text = text.replace(a, b)
dst.write_text(text, encoding="utf-8")
PY

"${PANDOC}" "${TMP_MD}" -o "${PDF_DIR}/pdf2.pdf" --pdf-engine=pdflatex \
  --resource-path="${ROOT}:${ROOT}/.." -V geometry:margin=1in
echo "Wrote ${PDF_DIR}/pdf2.pdf"
