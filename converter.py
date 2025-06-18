import pymupdf4llm
import pathlib
pdf_dir = pathlib.Path("../../books/PCM.pdf")
print(pdf_dir)
doc = pymupdf4llm.to_markdown(str(pdf_dir))
pathlib.Path("PCM_output.md").write_bytes(doc.encode())
# md_dir = pathlib.Path("converted_md")
# md_dir.mkdir(exist_ok=True)

# for pdf_file in pdf_dir.glob("*.pdf"):
#     print(pdf_file)
#     doc = pymupdf4llm.to_markdown(str(pdf_file))
#     text = doc.get_text()
#     print(text)
#     doc.close()
# print("Done")