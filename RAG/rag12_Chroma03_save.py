
import os
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

api_key = os.getenv("MONOROUTER_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("MONOROUTER_API_KEY 환경 변수를 설정해 주세요.")
api_key = api_key.strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

from glob import glob

path = 'c:/study/furiosa-tensorflow-keras/keras/_data/rag_data/'

# 폴더에서 텍스트 파일 목록 가져오기
txt_files = sorted(glob(os.path.join(path, '*.txt')))
print(txt_files)

# 01. 데이터 불러오기 (Document Loader)
# RAG를 위해 외부 텍스트 파일(삼성, 엔비디아 전망)을 읽어오는 로더를 세팅합니다.
# Document는 page_content와 meta_data가 있다.
# metadata 의 예로는 경로가 있다.
# 수업 필기: Document는 page_content와 meta_data가 있다.
# 보완: 정확한 속성명은 meta_data가 아니라 metadata이다.
#       Document는 주로 page_content에 실제 문서 내용, metadata에 출처/파일경로 등의 부가정보를 가진다.

data = []

for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    documents = loader.load()  # ★ 로더에 들어있는 텍스트를 실제 '문서 객체(Document)'로 읽어옵니다.
    data += documents  # 읽어온 문서들을 data 리스트에 합쳐줍니다.
    # 보완: data += documents는 data.extend(documents)와 비슷하게 리스트에 여러 Document를 추가한다.

print(data)
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)  # [8158, 2049, 1898]
# 보완: len(page_content)는 기본적으로 Token 수가 아니라 문자열의 길이를 센다.

# 02. 데이터 자르기 (Text Splitter)
# 문서를 통째로 넣으면 모델 용량 초과/검색 비효율이 발생하므로 적당한 크기(Chunk)로 쪼갭니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 한 조각(청크)에 들어갈 최대 글자 수 (300자씩 자르기)
    chunk_overlap=10,  # 앞뒤 조각이 10자씩 겹치게 잘라서 문맥이 끊기지 않게 함
    separators=["\n\n", "\n", " ", ""]  # 자르는 기준 우선순위 (문단 -> 줄 -> 띄어쓰기 순)
)
# 수업 필기: chunk_size=300, chunk_overlap=10
# 보완: 기존 주석에는 "100자씩 겹침"이라고 되어 있었지만 실제 값은 10이므로 10자가 맞다.
# 보완: RecursiveCharacterTextSplitter는 separator를 순서대로 시도하면서 가능한 자연스러운 위치에서 문서를 나눈다.

texts = text_splitter.split_documents(data)
# 수업 필기: split_documents()로 Document들을 Chunk 단위로 자른다.
# 보완: 반환값도 문자열 리스트가 아니라 list[Document]이다.
#       잘린 각 Chunk는 page_content와 metadata를 유지하는 Document 객체이다.

print("생성된 텍스트 청크수: ", len(texts))  # 59
print("각 청크의 길이: ", [len(text.page_content) for text in texts])

print("첫번째 청크의 내용: ", texts[0].page_content)
print("첫번째 청크의 길이: ", len(texts[0].page_content))
print("두번째 청크의 내용: ", texts[1].page_content)
print("두번째 청크의 길이: ", len(texts[1].page_content))



embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)



DB_PATH = 'c:/study/furiosa-tensorflow-keras/_db/Chroma12'
# 보완: 맨 위에서 import한 DB_PATH를 여기서 새 값으로 다시 할당하므로 이전 값은 사용되지 않는다.

ids = [
    hashlib.sha256(
        f"{doc.metadata.get('source', '')}\0{index}\0{doc.page_content}".encode("utf-8")
    ).hexdigest()
    for index, doc in enumerate(texts)
]

vector_store = Chroma.from_documents(
    documents=texts,
    ids=ids,  # 재실행 시 같은 청크를 같은 ID로 갱신
    embedding=embeddings,
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

# 디폴트값 k = 4이기 때문에 4개가 반환됨
# 수업 필기: similarity_search()의 디폴트 k=4이므로 관련 문서 4개가 반환된다.
# 보완: similarity_search()는 query를 임베딩한 뒤 Vector Store에서 가까운 Document들을 찾아 반환한다.
#       반환값은 벡터가 아니라 list[Document]이다.

print(f"검색 결과의 길이: {len(result)}")  # similarity_search()의 디폴트 k 값이 4라서 4

############################# Retriever #############################
############################# 검색기 #############################

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
