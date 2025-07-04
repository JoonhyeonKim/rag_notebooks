import pathlib
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === 0. 환경 설정 ===
load_dotenv()
md_dir = pathlib.Path("converted_md_for_ml")
faiss_path = pathlib.Path("faiss_db_for_ml/index.faiss")
cache_store = LocalFileStore("./cache_for_ml/")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding_model, cache_store, namespace=embedding_model.model
)

markdown_splitter = MarkdownHeaderTextSplitter([
    ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")
])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

# === 1. Markdown 로드 및 정규화 ===
def normalize(text: str) -> str:
    return " ".join(text.split())

all_texts = []
for md_file in sorted(md_dir.glob("*.md")):
    base_name = md_file.stem
    print(f"📄 Processing {md_file.name}...")
    md_text = md_file.read_text(encoding="utf-8")
    docs = markdown_splitter.split_text(md_text)
    for doc in docs:
        doc.metadata["source"] = base_name
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = normalize(chunk.page_content)
    all_texts.extend(chunks)

# === 2. FAISS 인덱스가 있다면 바로 로드 ===
# if faiss_path.exists():
#     print("📦 FAISS index found. Loading...")
#     db = FAISS.load_local("faiss_db_for_ml", cached_embedder, allow_dangerous_deserialization=True)

# else:
#     print("🧠 No FAISS index. Checking embedding cache...")

#     # 캐시 확인 후 임베딩 필요시 수행
#     def batch_iter(items, batch_size):
#         for i in range(0, len(items), batch_size):
#             yield items[i:i + batch_size]

#     cached_keys = list(cache_store.yield_keys(prefix=embedding_model.model))
#     if not cached_keys:
#         print("💾 No cached embeddings found. Embedding now...")
#         for batch in batch_iter([doc.page_content for doc in all_texts], 50):
#             _ = cached_embedder.embed_documents(batch)
#     else:
#         print(f"✅ {len(cached_keys)} cached embeddings found. Skipping embedding.")

#     # FAISS 인덱스 생성
#     print("🛠️ Creating FAISS index...")
#     db = FAISS.from_documents(all_texts, cached_embedder)
#     db.save_local("faiss_db_for_ml")


# === 2. FAISS 인덱스가 있다면 바로 로드 ===
if faiss_path.exists():
    print("📦 FAISS index found. Loading...")
    db = FAISS.load_local("faiss_db_for_ml", cached_embedder, allow_dangerous_deserialization=True)

    # 기존에 인덱싱된 소스 목록 추출
    existing_sources = set()
    try:
        # FAISS 내부 문서 일부를 통해 metadata source 목록 수집
        for doc in db.similarity_search("dummy", k=100):
            if "source" in doc.metadata:
                existing_sources.add(doc.metadata["source"])
    except Exception:
        print("⚠️ Couldn't extract existing sources from index.")

    # 새로 추가할 문서만 추려내기
    new_docs = [doc for doc in all_texts if doc.metadata.get("source") not in existing_sources]
    print(f"🆕 Found {len(new_docs)} new documents to add.")

    if new_docs:
        db_new = FAISS.from_documents(new_docs, cached_embedder)
        db.merge_from(db_new)
        db.save_local("faiss_db_for_ml")
        print("✅ FAISS updated with new documents.")
    else:
        print("✅ No new documents to add.")

else:
    print("🧠 No FAISS index. Creating from scratch...")

    def batch_iter(items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    cached_keys = list(cache_store.yield_keys(prefix=embedding_model.model))
    if not cached_keys:
        print("💾 No cached embeddings found. Embedding now...")
        for batch in batch_iter([doc.page_content for doc in all_texts], 50):
            _ = cached_embedder.embed_documents(batch)
    else:
        print(f"✅ {len(cached_keys)} cached embeddings found. Skipping embedding.")

    db = FAISS.from_documents(all_texts, cached_embedder)
    db.save_local("faiss_db_for_ml")
    print("✅ FAISS index created and saved.")


# === 3. Retriever 설정 ===
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))
compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

# === 4. 질문 입력 및 질의 확장 ===
question = "PC 알고리즘에 대해 알려줘"
llm = ChatOpenAI(model="gpt-4o", temperature=0)

query_prompt = PromptTemplate.from_template("""
너는 질문을 더 잘 검색되도록 다양한 표현으로 바꾸는 시스템이야.
다음 질문을 3가지 다른 표현으로 바꾸어 줄바꿈으로 출력해줘.

질문: {question}
""")
query_chain = query_prompt | llm | StrOutputParser()
expanded_queries = query_chain.invoke({"question": question}).strip().split("\n")
print("\n🧠 Expanded Queries:", expanded_queries)

# === 5. 검색 및 압축 결과 확인 ===
query_to_docs = {}
for q in expanded_queries:
    docs = compression_retriever.invoke(q)
    query_to_docs[q] = docs

    # 🔍 검색 결과 출력
    print(f"\n🔎 Retrieved for query: {q}")
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        print(f"[출처: {source}]\n{doc.page_content.strip()[:300]}...\n")

# === 6. RAG 프롬프트 구성 ===
def format_query_docs(query_to_docs):
    blocks = []
    for q, docs in query_to_docs.items():
        block = [f"🔎 Query: {q}"]
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            block.append(f"[출처: {source}]\n{doc.page_content.strip()}")
        blocks.append("\n".join(block))
    return "\n\n---\n\n".join(blocks)

formatted_context = format_query_docs(query_to_docs)

rag_prompt = f"""
다음은 원 질문에 대해 다양한 방식으로 재작성된 질의와 그에 대한 검색 결과 요약이다:

## Context
{formatted_context}

## Instructions
기존의 훈련 정보를 무시하고 이 정보만을 바탕으로 다음 질문에 정확하고 포괄적으로 답하라.
비전문가가 이해할 수 있도록 용어를 쉽게 풀고 예시를 들어 설명하라.

## Question
Q: {question}
"""

# === 7. 최종 응답 생성 ===
answer = llm.invoke(rag_prompt)

print("\n📌 RAG 결과:")
print(answer)
