import pathlib
import pymupdf4llm

pdf_dir = pathlib.Path("../../books/")
md_dir = pathlib.Path("converted_md_for_ml")
md_dir.mkdir(exist_ok=True)

for pdf_file in pdf_dir.glob("*.pdf"):
    base_name = pdf_file.stem
    md_file = md_dir / f"{base_name}.md"

    if not md_file.exists():
        print(f"Converting {pdf_file.name} to markdown...")
        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        md_file.write_text(md_text, encoding="utf-8")
    else:
        print(f"{md_file.name} already exists. Skipping.")
