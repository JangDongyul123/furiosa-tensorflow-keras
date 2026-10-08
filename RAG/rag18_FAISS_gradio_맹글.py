import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

api_key = os.getenv("MONOROUTER_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("MONOROUTER_API_KEY 환경 변수를 설정해 주세요.")
api_key = api_key.strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
NO_ANSWER = "주어진 정보로는 답변할 수 없습니다."

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = "./_db/Faiss17"
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 임계값은 현재 DB에서 관련/무관 질문의 점수를 확인해 조정하세요.
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.05},
)

model = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 검색된 삼성전자·엔비디아 전망 문서만 보고 답하는 챗봇이다.
답변의 근거는 반드시 컨텍스트 안에 있어야 하며, 일반 지식이나 추측을 덧붙이지 마라.
질문과 관련된 근거가 없으면 정확히 '{no_answer}'만 출력하라.
이전 대화는 후속 질문의 지시 대상(예: '그 회사')을 파악할 때만 사용하라.

이전 대화:
{history}

컨텍스트:
{context}"""),
    ("human", "{input}"),
])
doc_chain = create_stuff_documents_chain(model, prompt)


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content or "")


def history_entries(history):
    """Gradio messages 형식과 구형 (user, assistant) 튜플 형식을 모두 정리합니다."""
    entries = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role", "user")
            entries.append((role, content_to_text(item.get("content", ""))))
        elif isinstance(item, (list, tuple)):
            if len(item) > 0 and item[0] is not None:
                entries.append(("user", content_to_text(item[0])))
            if len(item) > 1 and item[1] is not None:
                entries.append(("assistant", content_to_text(item[1])))
    return entries


def answer_invoke(message, history):
    entries = history_entries(history)
    previous_user = next(
        (content for role, content in reversed(entries) if role == "user"),
        "",
    )
    search_query = f"{previous_user} {message}".strip()
    docs = retriever.invoke(search_query)
    if not docs:
        return NO_ANSWER

    history_text = "\n".join(
        f"{'사용자' if role == 'user' else '챗봇'}: {content}"
        for role, content in entries[-6:]
    ) or "(없음)"

    answer = doc_chain.invoke({
        "context": docs,
        "input": message,
        "history": history_text,
        "no_answer": NO_ANSWER,
    })

    sources = sorted({
        os.path.basename(doc.metadata["source"])
        for doc in docs
        if doc.metadata.get("source")
    })
    if sources:
        return f"{answer}\n\n📄 출처: {', '.join(sources)}"
    return answer


demo = gr.ChatInterface(
    fn=answer_invoke,
    title="삼성전자 · 엔비디아 RAG 챗봇",
    description="저장된 FAISS 문서의 근거로 답하고, 관련 문서가 없으면 답변을 보류합니다.",
    examples=[
        "삼성전자의 주요 사업은 뭐야?",
        "엔비디아의 데이터센터 사업은 어때?",
        "그 회사의 위험 요인은?",
        "김치찌개 끓이는 법 알려줘",
    ],
)
demo.launch()
