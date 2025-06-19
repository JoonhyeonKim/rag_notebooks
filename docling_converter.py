from langchain_docling import DoclingLoader
import pathlib
pdf_path = pathlib.Path("../../books/PCM.pdf")
print(pdf_path)

loader = DoclingLoader(file_path=pdf_path)
# docs = loader.load()
# print(docs)

doc_iter = loader.lazy_load()
for doc in doc_iter:
    print(doc)