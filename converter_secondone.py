import pymupdf4llm
import pathlib

pdf_path = pathlib.Path("../../books/PCM.pdf")
print(pdf_path)

# PDF → 페이지 단위 dict 리스트
doc = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=True)

lines = []

for page in doc:
    meta = page.get("metadata", {})
    page_number = meta.get("page_number", "N/A")
    page_count = meta.get("page_count", "N/A")
    file_path = meta.get("file_path", "N/A")
    
    text = page.get("text", "")

    lines.append(
        f"## Page {page_number}\n\n"
        f"**File:** {file_path}\n\n"
        f"**Page Number:** {page_number} / {page_count}\n\n"
        f"### Content:\n{text}\n\n---\n"
    )

# 파일로 저장
output = "\n".join(lines)
pathlib.Path("PCM_output.md").write_bytes(output.encode())
