# 12-4 카피

import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

api_key = os.getenv("MONOROUTER_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("MONOROUTER_API_KEY 환경 변수를 설정해 주세요.")
api_key = api_key.strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = 'c:/study/furiosa-tensorflow-keras/_db/Chroma12'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma12'
)
# 수업 필기: Chunk들을 임베딩하여 Chroma Vector Store에 저장한다.
# 흐름: Document → Chunk → Embedding Vector → Chroma Vector Store

print(f"벡터 저장소에 저장된 문서 수: {vector_store._collection.count()}")
# 보완: _collection처럼 이름 앞에 _가 붙은 속성은 내부 구현용(private 성격)이다.
#       실습에서 확인용으로 사용할 수 있지만 라이브러리 버전에 따라 변경될 수 있다.

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    # 0.05는 시작값입니다. 실제 DB의 관련/무관 질의 점수를 비교해 조정하세요.
    search_kwargs={"k": 2, "score_threshold": 0.05}
)


################ 모델 연결 #################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens = 1000,
    api_key = api_key,
    base_url = base_url,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

NO_ANSWER = "주어진 정보로는 답변할 수 없습니다."
prompt = ChatPromptTemplate.from_messages([
    ("system", """검색된 컨텍스트에 있는 정보만 사용해 답변하세요.
질문과 관련된 근거가 컨텍스트에 없으면 다른 설명 없이 정확히 '{no_answer}'만 출력하세요.
추측하거나 일반 지식을 보태지 마세요.

이전 대화는 후속 질문의 지시 대상을 확인할 때만 사용하세요.
이전 대화:
{history}

컨텍스트:
{context}"""),
    ("human", "{input}"),
])

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
# prompt | model

# ---------------------------------------------------------------------------
# Gradio 웹 인터페이스 구현
# ---------------------------------------------------------------------------
import gradio as gr

def answer_invoke(message, history):
    def content_to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return str(content or "")

    history = history or []
    previous_user = ""
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            previous_user = content_to_text(item.get("content", ""))
            break
        if isinstance(item, (list, tuple)) and item:
            previous_user = content_to_text(item[0])
            break

    # 후속 질문의 회사 지시어를 검색할 때 직전 사용자 질문으로 보완합니다.
    search_query = f"{previous_user} {message}".strip()
    docs = retriever.invoke(search_query)
    if not docs:
        return NO_ANSWER

    if history and isinstance(history[-1], dict):
        recent_history = history[-6:]
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {content_to_text(item.get('content', ''))}"
            for item in recent_history
            if isinstance(item, dict)
        ) or "(없음)"
    else:
        history_text = "\n".join(
            f"사용자: {content_to_text(pair[0])}\n챗봇: {content_to_text(pair[1])}"
            for pair in history[-3:]
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ) or "(없음)"

    return docu_chain.invoke({
        "context": docs,
        "input": message,
        "history": history_text,
        "no_answer": NO_ANSWER,
    })

# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn = answer_invoke, title = "동율채팅")

# Gradio 실행
demo.launch()
