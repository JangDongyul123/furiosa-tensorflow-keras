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

query = "삼성전자의 창업주는 누구인가요?"

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    # 0.05는 시작값입니다. 실제 DB의 관련/무관 질의 점수를 비교해 조정하세요.
    search_kwargs={"k": 2, "score_threshold": 0.05}
)
# 수업 필기: Retriever = 검색기
# 보완: Vector Store 자체를 LangChain에서 사용할 수 있는 Retriever 형태로 변환한다.
#       search_kwargs={"k": 2} → 질문과 관련성이 높은 Document 2개를 검색한다.
#       as_retriever()의 기본 search_type은 similarity이다.
print("============================================")
print("retriever: ", retriever)
print("============================================")

# aaa = retriever.invoke(query)
# # 수업 필기: Retriever도 invoke()로 실행한다.
# # 보완: query 문자열을 Retriever에 전달하면 관련 Document 목록을 반환한다.

# print(f"검색된 관련 문서 수: {len(aaa)}")
# print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}...")
# # page_content[:50] → 첫 번째 검색 결과의 앞 50글자만 출력


################ 모델 연결 #################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens = 1000,
    api_key = api_key,
    base_url = base_url,
)

# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# """

# response = model.invoke(query_with_context)
# print("model의 응답: ", response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

NO_ANSWER = "주어진 정보로는 답변할 수 없습니다."
prompt = ChatPromptTemplate.from_messages([
    ("system", """검색된 컨텍스트에 있는 정보만 사용해 답변하세요.
질문과 관련된 근거가 컨텍스트에 없으면 다른 설명 없이 정확히 '{no_answer}'만 출력하세요.
추측하거나 일반 지식을 보태지 마세요.

컨텍스트:
{context}"""),
    ("human", "{input}"),
])

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
# prompt | model

rag_chain = create_retrieval_chain(retriever, docu_chain)
# 검색 | docu_chain

# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input": query, "no_answer": NO_ANSWER})

print("response: ", response)
print("================== keys() =====================")
print(response.keys())
print("================= context =====================")
print(response['context'][0].page_content)
print("================== answer =====================")
print(response['answer'])

