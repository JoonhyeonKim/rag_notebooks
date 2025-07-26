import pathlib, re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────
text_dir   = pathlib.Path("../../sacred_text_/")
faiss_root = "faiss_db_for_religious"
faiss_path = pathlib.Path(f"{faiss_root}/index.faiss")

# ── 1) DirectoryLoader: 폴더 내 모든 .txt 로드 ─────────────
loader = DirectoryLoader(
    path=str(text_dir),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs_raw = loader.load()                     # List[Document]

# ── 2) 메타데이터 보강 (파일별 앞부분 5줄 스캔) ─────────────
def enrich_meta(doc):
    head = "\n".join(doc.page_content.splitlines()[:5]).lower()
    if "translated by" in head:
        doc.metadata["translator"] = re.search(
            r"translated by\s*([^\n]+)", head).group(1).strip()
    if "sacred-texts" in head:
        doc.metadata["source_site"] = "sacred-texts.com"
    doc.metadata["title"] = pathlib.Path(doc.metadata["source"]).stem
    return doc

docs_raw = [enrich_meta(d) for d in docs_raw]

# ── 3) Chunk split ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
docs = splitter.split_documents(docs_raw)

# ── 4) 임베딩 + 캐시 ────────────────────────────────────────
cache_store     = LocalFileStore("./cache_religious/")
embed_model     = OpenAIEmbeddings(model="text-embedding-3-large")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embed_model, cache_store, namespace=embed_model.model
)

# ── 5) FAISS 인덱스 생성·업데이트 ──────────────────────────
if faiss_path.exists():
    db = FAISS.load_local(faiss_root, cached_embedder,
                          allow_dangerous_deserialization=True)
    known = {d.metadata.get("title") for d in db.similarity_search("dummy", k=50)}
    new_docs = [d for d in docs if d.metadata["title"] not in known]
    if new_docs:
        db.merge_from(FAISS.from_documents(new_docs, cached_embedder))
        db.save_local(faiss_root)
        print(f"✅ {len(new_docs)} new docs added.")
else:
    db = FAISS.from_documents(docs, cached_embedder)
    db.save_local(faiss_root)
    print("✅ FAISS index created.")

# 이제 db.as_retriever() 로 바로 검색·RAG에 활용 가능