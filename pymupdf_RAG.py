import pymupdf4llm
import pathlib
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()


# === 1. PDF 처리 ===
pdf_dir = pathlib.Path("../pydantic-with-langgraph-agent/identity_laws_/")
md_dir = pathlib.Path("converted_md")
md_dir.mkdir(exist_ok=True)

cache_store = LocalFileStore("./cache/")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding_model, cache_store, namespace=embedding_model.model
)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

all_texts = []

for pdf_file in pdf_dir.glob("*.pdf"):
    base_name = pdf_file.stem
    md_file = md_dir / f"{base_name}.md"

    if not md_file.exists():
        print(f"Converting {pdf_file.name} to markdown...")
        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        md_file.write_text(md_text, encoding="utf-8")
    else:
        print(f"Markdown already exists for {pdf_file.name}. Skipping.")
        md_text = md_file.read_text(encoding="utf-8")

    docs = markdown_splitter.split_text(md_text)
    for doc in docs:
        doc.metadata["source"] = base_name
    chunks = text_splitter.split_documents(docs)
    all_texts.extend(chunks)

# === 2. 임베딩 캐시 처리 ===
def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

if not any(cache_store.yield_keys(prefix=embedding_model.model)):
    print("No embeddings in cache. Creating and caching embeddings...")
    all_contents = [doc.page_content for doc in all_texts]
    for batch in batch_iter(all_contents, batch_size=50):
        _ = cached_embedder.embed_documents(batch)
else:
    print("Embeddings already cached. Skipping embedding step.")

# === 3. FAISS DB 저장 ===
db = FAISS.from_documents(all_texts, cached_embedder)
db.save_local("faiss_db")
print("✅ Done. FAISS DB saved.")

# === 4. 질의 확장 ===
question = "개인정보가 유출되었을때 받을 수 있는 최대 금액??"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query_prompt = PromptTemplate.from_template("""
너는 질문을 더 잘 검색되도록 다양한 표현으로 바꾸는 시스템이야.
다음 질문을 3가지 다른 표현으로 바꾸어 줄바꿈으로 출력해줘.

질문: {question}
""")
query_chain = query_prompt | llm | StrOutputParser()
expanded_queries = query_chain.invoke({"question": question}).strip().split("\n")
print(f"📌 Expanded queries: {expanded_queries}")

# === 5. 압축 검색 설정 ===
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4})
compressor = LLMChainExtractor.from_llm(OpenAI(model="gpt-4o-mini", temperature=0))
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# === 6. 질의별 문서 검색 및 정리 ===
query_to_docs = {}
for q in expanded_queries:
    docs = compression_retriever.invoke(q)
    query_to_docs[q] = docs
    print(docs)

# === 7. AG용 Prompt 구성 ===
def format_query_docs(query_to_docs):
    blocks = []
    for q, docs in query_to_docs.items():
        block = [f"🔎 Query: {q}"]
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            block.append(f"[출처: {source}]\n{content}")
        blocks.append("\n".join(block))
    return "\n\n---\n\n".join(blocks)

formatted_context = format_query_docs(query_to_docs)

rag_prompt = f"""
다음은 원 질문에 대해 다양한 방식으로 재작성된 질의와 그에 대한 검색 결과 요약이다:
## Context
{formatted_context}
## Instructions
기존의 훈련 정보를 무시하고 이 정보만을 바탕으로 다음 질문에 정확하고 포괄적으로 답하라
사용자는 법률 전문가가 아니며, 이 정보를 바탕으로 답변은 비전문가가 이해할 수 있도록 작성되어야 한다.
어려운 용어는 풀어서 쓰고 예시를 들어 설명하라.
## Question
Q: {question}
"""

# === 8. AG 실행 ===
llm_ag = ChatOpenAI(model="gpt-4o-mini", temperature=0)
answer = llm_ag.invoke(rag_prompt)

print("\n📌 RAG 결과:")
print(answer)