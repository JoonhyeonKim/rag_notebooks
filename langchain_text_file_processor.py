"""
sacred_text_의 .txt → chunk → 임베딩 → FAISS
토큰 합계 28만 이하로 배치 분할 (OpenAI 임베딩 한도 30만 대비)
"""

import pathlib, re, tiktoken
from typing import List, Generator
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ── 설정 ───────────────────────────────────────────────────
load_dotenv()
TEXT_DIR   = pathlib.Path("../../sacred_text_/")
FAISS_ROOT = "faiss_db_for_religious"
FAISS_PATH = pathlib.Path(f"{FAISS_ROOT}/index.faiss")
CACHE_DIR  = "./cache_religious/"
TOK_LIMIT  = 280_000                        # 배치당 토큰 상한

# ── 1. 문서 로드 ───────────────────────────────────────────
loader = DirectoryLoader(
    path=str(TEXT_DIR),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs_raw: List[Document] = loader.load()

# ── 2. 메타데이터 보강 ────────────────────────────────────
def enrich_meta(doc: Document) -> Document:
    head = "\n".join(doc.page_content.splitlines()[:5]).lower()
    if "translated by" in head:
        m = re.search(r"translated by\s*([^\n]+)", head)
        if m:
            doc.metadata["translator"] = m.group(1).strip()
    if "sacred-texts" in head:
        doc.metadata["source_site"] = "sacred-texts.com"
    doc.metadata["title"] = pathlib.Path(doc.metadata.get("source", "")).stem
    return doc

docs_raw = [enrich_meta(d) for d in docs_raw]

# ── 3. chunk split ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
docs: List[Document] = splitter.split_documents(docs_raw)
print(f"📄 {len(docs_raw)} files → {len(docs)} chunks")

# ── 4. 임베딩 객체 + 캐시 ───────────────────────────────────
cache_store     = LocalFileStore(CACHE_DIR)
embed_model     = OpenAIEmbeddings(model="text-embedding-3-large")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embed_model, cache_store, namespace=embed_model.model
)

# ── 5. 토큰 한도 기반 배치 제너레이터 ─────────────────────
enc = tiktoken.get_encoding("cl100k_base")

def token_batches(sequence: List[Document], max_tokens: int) -> Generator[List[Document], None, None]:
    batch, tok_sum = [], 0
    for doc in sequence:
        tok_len = len(enc.encode(doc.page_content))
        if batch and tok_sum + tok_len > max_tokens:
            yield batch
            batch, tok_sum = [], 0
        batch.append(doc)
        tok_sum += tok_len
    if batch:
        yield batch

# ── 6. FAISS 인덱스 생성/업데이트 ──────────────────────────
if FAISS_PATH.exists():
    print("📦 기존 인덱스 로드")
    db = FAISS.load_local(FAISS_ROOT, cached_embedder,
                          allow_dangerous_deserialization=True)

    known_titles = {d.metadata.get("title") for d in db.similarity_search("dummy", k=100)}
    new_docs = [d for d in docs if d.metadata.get("title") not in known_titles]
    print(f"🆕 새 chunks: {len(new_docs)}")

    for i, batch in enumerate(token_batches(new_docs, TOK_LIMIT), 1):
        db.add_documents(batch, embedding=cached_embedder)
        print(f"  • added batch {i} ({len(batch)} chunks)")
    db.save_local(FAISS_ROOT)
    print("✅ 인덱스 업데이트 완료")

else:
    print("🧠 새 인덱스 생성…")
    batch_iter = token_batches(docs, TOK_LIMIT)
    first = next(batch_iter)
    db = FAISS.from_documents(first, cached_embedder)
    print(f"  • 초기 {len(first)} chunks 삽입")

    for i, batch in enumerate(batch_iter, 2):
        db.add_documents(batch, embedding=cached_embedder)
        if i % 20 == 0:
            print(f"  • batch {i} processed")
    db.save_local(FAISS_ROOT)
    print("✅ 새 인덱스 생성 완료")

# ── 7. 간단 테스트 ─────────────────────────────────────────
# if __name__ == "__main__":
#     retriever = db.as_retriever(k=3)
#     query = "Who translated the works of Sri Sankaracharya?"
#     for doc in retriever.invoke(query):
#         print(f"\n[{doc.metadata.get('title')}] {doc.page_content[:200]}…")
