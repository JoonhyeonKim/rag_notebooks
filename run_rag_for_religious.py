import pathlib
from dotenv import load_dotenv
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

faiss_path = pathlib.Path("faiss_db_for_religious/index.faiss")
cache_store = LocalFileStore("./cache_religious/")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding_model, cache_store, namespace=embedding_model.model
)


db = FAISS.load_local("faiss_db_for_religious", cached_embedder, allow_dangerous_deserialization=True)

# === 1. Retriever 설정 ===
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4})
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))
compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

# === 2. 질문 입력 및 질의 확장 ===
question = "아즈텍의 신앙에 대해 알려줘."
llm = ChatOpenAI(model="gpt-4o", temperature=0)
## HyDE Prompt
query_prompt = PromptTemplate.from_template("""
You act as a expert about the topic of user query.
Translate the question into English.
Take a deep breath and think carefully.
Now generate 3 different hypothetical answers for the question.

질문: {question}
""")
query_chain = query_prompt | llm | StrOutputParser()
expanded_queries = query_chain.invoke({"question": question}).strip().split("\n")
print("\n🧠 Expanded Queries:", expanded_queries)

# === 3. 검색 및 압축 결과 확인 ===
query_to_docs = {}
for q in expanded_queries:
    docs = compression_retriever.invoke(q)
    query_to_docs[q] = docs

    # 🔍 검색 결과 출력
    print(f"\n🔎 Retrieved for query: {q}")
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        print(f"[출처: {source}]\n{doc.page_content.strip()[:300]}...\n")

# === 4. RAG 프롬프트 구성 ===
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
- 기존의 훈련 정보를 무시하고 이 정보만을 바탕으로 다음 질문에 정확하고 포괄적으로 답하라.
- 비전문가가 이해할 수 있도록 용어를 쉽게 풀고 예시를 들어 설명하라.
- 질문자와 같은 언어로 대답하라.

## Question
Q: {question}
"""

# === 5. 최종 응답 생성 ===
answer = llm.invoke(rag_prompt)

print("\n📌 RAG 결과:")
print(answer)
