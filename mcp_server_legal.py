from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# FastMCP 서버 초기화
mcp = FastMCP(
    "LegalAssistant",
    instructions="법률 문서와 판례를 검색하고 분석하는 도구입니다.",
    host="0.0.0.0",
    port=8005,
    settings={"initialization_timeout": 10.0}
)

# ChromaDB 설정
PERSIST_DIRECTORY = "./chroma_db_legal_precedents"

def get_vectorstore():
    """ChromaDB 벡터 저장소 초기화"""
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    )

@mcp.tool()
async def search_legal_precedents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """법률 질의에 관련된 판례 검색"""
    logging.info(f"🔍 [search_legal_precedents] Query: {query}")
    vectorstore = get_vectorstore()
    results = vectorstore.max_marginal_relevance_search(query, k=top_k, lambda_mult=0.8)
    logging.info(f"✅ [search_legal_precedents] Found {len(results)} results")
    return [{
        "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
        "metadata": doc.metadata
    } for doc in results]

@mcp.tool()
async def analyze_legal_situation(situation: str) -> Dict[str, Any]:
    """법률 상황 분석"""
    logging.info(f"🧠 [analyze_legal_situation] Situation: {situation}")
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(situation, k=3)
    logging.info(f"📊 [analyze_legal_situation] Analyzed {len(results)} documents")
    return {
        "precedents": [doc.metadata for doc in results],
        "related_laws": list(set(doc.metadata.get("law_code", "") for doc in results))
    }

if __name__ == "__main__":
    logging.info("⚙️ 법률 RAG MCP 서버 시작 중...")
    mcp.run(transport="stdio")
    logging.info("🛑 서버 종료")
