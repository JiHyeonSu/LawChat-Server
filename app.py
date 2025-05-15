import streamlit as st
import asyncio
import nest_asyncio
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

import uuid
import json
import os

# 비동기 이벤트 루프 설정
nest_asyncio.apply()
load_dotenv()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


MCP_CONFIG = {
    "legal_rag": {
        "command": "python",
        "args": ["./mcp_server_legal.py"],
        "transport": "stdio"
    }
}

# 페이지 설정
st.set_page_config(page_title="법률 RAG 시스템", layout="wide")
st.title("⚖️ 법률 RAG 시스템")

# 시스템 프롬프트
SYSTEM_PROMPT = """
[기존 프롬프트 유지]
**반드시 다음 도구를 사용하세요:**
- search_legal_precedents: 판례 검색
- analyze_legal_situation: 법률 상황 분석
"""


class LegalAssistant:
  def __init__(self):
    self.client = None
    self.agent = None
    self.tools = []

  async def initialize(self):
    try:
      # MCP 클라이언트 초기화
      self.client = MultiServerMCPClient(MCP_CONFIG)
      await self.client.__aenter__()

      # 도구 목록 가져오기
      self.tools = self.client.get_tools()
      if not self.tools:
        raise ValueError("MCP 서버에서 도구를 찾을 수 없습니다")

      # 모델 초기화
      model = ChatAnthropic(
          model="claude-3-7-sonnet-latest",
          temperature=0.1,
          max_tokens=64000
      )

      # ReAct 에이전트 생성
      self.agent = create_react_agent(
          model,
          self.tools,
          prompt=SYSTEM_PROMPT,
          checkpointer=MemorySaver()
      )
      return True

    except Exception as e:
      st.error(f"초기화 실패: {str(e)}")
      return False


async def process_query(assistant, query, progress_callback):
  try:
    # 1단계: 질문 분석
    progress_callback(10, "질문 분석 중...")

    # 2단계: 도구 실행
    progress_callback(30, "판례 검색 중...")
    response = await assistant.agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 100,
                "thread_id": st.session_state.thread_id
                }
    )

    # 3단계: 결과 종합
    progress_callback(80, "답변 생성 중...")

    return response["messages"][-1].content

  except asyncio.TimeoutError:
    raise Exception("응답 시간 초과 (60초 이내에 처리되지 않았습니다)")
  except Exception as e:
    raise e


def main():
  # 세션 상태 초기화
  if "assistant" not in st.session_state:
    st.session_state.assistant = LegalAssistant()
    st.session_state.initialized = False

  # 사이드바 설정
  with st.sidebar:
    if st.button("🔄 시스템 초기화", type="primary"):
      with st.spinner("MCP 서버 연결 중..."):
        if asyncio.run(st.session_state.assistant.initialize()):
          st.session_state.initialized = True
          st.success("✅ 초기화 완료")
        else:
          st.error("❌ 초기화 실패")

  # 초기화 확인
  if not st.session_state.initialized:
    st.warning("⚠️ 먼저 사이드바에서 시스템 초기화를 진행해주세요")
    return

  # 사용자 입력 처리
  user_input = st.chat_input("법률 질문을 입력하세요")
  if user_input:
    # 진행 상태 관리
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent, message):
      progress_bar.progress(percent)
      status_text.markdown(f"**진행 상태**: {message}")

    # 질문 처리
    with st.chat_message("user"):
      st.markdown(user_input)

    with st.chat_message("assistant"):
      try:
        response = asyncio.run(
            process_query(
                st.session_state.assistant,
                user_input,
                update_progress
            )
        )
        st.markdown(response)

      except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        st.session_state.initialized = False


if __name__ == "__main__":
  main()
