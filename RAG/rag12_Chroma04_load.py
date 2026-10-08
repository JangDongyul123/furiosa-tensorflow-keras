
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
# 보완: 맨 위에서 import한 DB_PATH를 여기서 새 값으로 다시 할당하므로 이전 값은 사용되지 않는다.

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
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이: {len(result)}")  # similarity_search()의 디폴트 k 값이 4라서 4

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
# 수업 필기: Retriever = 검색기
# 보완: Vector Store 자체를 LangChain에서 사용할 수 있는 Retriever 형태로 변환한다.
#       search_kwargs={"k": 2} → 질문과 관련성이 높은 Document 2개를 검색한다.
#       as_retriever()의 기본 search_type은 similarity이다.

print(retriever)

aaa = retriever.invoke(query)
# 수업 필기: Retriever도 invoke()로 실행한다.
# 보완: query 문자열을 Retriever에 전달하면 관련 Document 목록을 반환한다.

print(f"검색된 관련 문서 수: {len(aaa)}")
print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}...")
# page_content[:50] → 첫 번째 검색 결과의 앞 50글자만 출력
