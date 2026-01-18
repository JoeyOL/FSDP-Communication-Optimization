from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--out", required=True, help="Path to output .txt")
    parser.add_argument("--max_pages", type=int, default=0, help="0 means all pages")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pypdf import PdfReader
    except Exception as e:
        raise SystemExit(f"Failed to import pypdf. Install it first. Error: {e}")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else total_pages
    max_pages = min(max_pages, total_pages)

    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# Source: {pdf_path.name}\n")
        f.write(f"# Pages: {total_pages}\n")
        f.write("\n")
        for i in range(max_pages):
            text = reader.pages[i].extract_text() or ""
            f.write(f"\n\n--- Page {i+1} ---\n")
            f.write(text)

    print(f"Wrote: {out_path}")
    print(f"Pages extracted: {max_pages}/{total_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
