# 11-1 카피

# 개인공부 보완: ModelCheckpoint, RLR 모든 옵션을 상세하게 공부하자

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

api_key = os.getenv("MONOROUTER_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("MONOROUTER_API_KEY 환경 변수를 설정해 주세요.")
api_key = api_key.strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

# 01. 데이터 불러오기 (Document Loader)
# RAG를 위해 외부 텍스트 파일(삼성, 엔비디아 전망)을 읽어오는 로더를 세팅합니다.
path = 'c:/study/furiosa-tensorflow-keras/keras/_data/rag_data/'
loader1 = TextLoader(path + "samsung_outlook.txt", encoding='utf-8')
loader2 = TextLoader(path + "nvidia_outlook.txt", encoding='utf-8')

# 02. 데이터 자르기 (Text Splitter)
# RAG는 Chunk 단위로, 모델 입력과 학습은 Token 단위로 벡터화 한다.
# 수업 필기: RAG는 Chunk 단위로, 모델 입력과 학습은 Token 단위로 벡터화 한다.
# 보완: RAG에서는 문서를 Chunk 단위로 나눈 뒤 각 Chunk 전체를 임베딩 모델에 넣어 하나의 벡터로 변환한다.
# 보완: LLM은 입력을 Token 단위로 처리하지만, RAG에서 Token 하나마다 임베딩 벡터 하나를 만드는 것은 아니다.

# Chunk는 Token보다 더 큰 단위이다. RAG에서 Chunk 단위로 사용하는 이유는
# Token 단위일 때보다 속도가 빠르고, 문맥 파악이 수월해서 그렇다.
# 수업 필기: Chunk는 Token보다 더 큰 단위이며 Token 단위보다 속도가 빠르고 문맥 파악이 수월하다.
# 보완: Chunk는 여러 Token을 포함한 텍스트 덩어리이다. Chunk 단위로 검색하면 개별 Token보다 문맥과 의미를 보존하기 쉽고 검색할 벡터 수도 줄어든다.

# 문서를 통째로 넣으면 모델 용량 초과/검색 비효율이 발생하므로 적당한 크기(Chunk)로 쪼갭니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    # 한 조각(청크)에 들어갈 최대 글자 수 (300자씩 자르기)
    # 보완: 기본 length_function이 len이므로 여기서 300은 Token 300개가 아니라 문자열 길이 300을 의미한다.
    chunk_overlap=100,
    # 앞뒤 조각이 100자씩 겹치게 잘라서 문맥이 끊기지 않게 함
    separators=["\n\n", "\n", " ", ""]
    # 자르는 기준 우선순위 (문단 -> 줄 -> 띄어쓰기 순)
)

# 수업 필기
# 150 \n\n
# 140 \n\n
# 130
# 이면 290에서 끊김
# 보완: RecursiveCharacterTextSplitter는 무조건 정확히 300에서 자르는 것이 아니라
# separators 우선순위에 따라 chunk_size를 넘지 않는 자연스러운 위치에서 자르려고 한다.
# 따라서 150 + 140 = 290까지 합쳐지고 다음 내용을 붙이면 300을 넘는 경우 290 부근에서 Chunk가 끝날 수 있다.

# 실제로 파일을 읽어오면서(load) 동시에 위 설정대로 자릅니다(split).
# 결과물은 조각(청크)들의 리스트(List)가 됩니다.
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

# 문서 개수 확인
print(split_doc1)
print(len(split_doc1), len(split_doc2))

# 9 -> 1536개짜리 배열이 9개 있는 것
# 수업 필기: 9 -> 1536개짜리 배열이 9개 있는 것
# 보완: 이 시점의 split_doc1에는 아직 1536차원 임베딩 벡터가 아니라 Document 객체 9개가 들어 있다.
# 즉 len(split_doc1) == 9라면 Chunk(Document)가 9개 있다는 뜻이다.
# 이후 Chroma에 저장하는 과정에서 각 Chunk가 임베딩되어 1536차원 벡터 하나로 변환된다.

# [, , , , , ] 이런 요소가 1536개짜리 배열이 9개 있는 것
# 청크 한개를 나타내는 게 1536개짜리 배열로 나타내는 것.
# 즉 청크 한개가 벡터가 된다. (벡터화)
# 벡터 스토어에 넣는다.
# 수업 필기: 청크 한 개가 1536개 숫자로 이루어진 벡터 한 개가 되어 Vector Store에 들어간다.
# 보완: text-embedding-3-small의 기본 출력 차원이 1536이므로 Chunk 9개를 임베딩하면 개념적으로 1536차원 벡터 9개가 생성된다.
# 보완: Vector Store에는 벡터뿐 아니라 원본 텍스트(page_content), metadata 등도 함께 저장될 수 있다.

from langchain_openai import OpenAIEmbeddings

# 텍스트를 1536개의 숫자로 이루어진 벡터로 변환하는 '임베딩 모델' 객체를 생성합니다.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 빠르고 저렴하며 성능이 좋은 기본 임베딩 모델
    api_key=api_key,                # API 인증 키
    base_url=base_url,              # 통신할 서버의 API 주소
    # dimensions=5                  # (참고) 출력할 벡터 차원을 줄일 때 사용
)

db = FAISS.from_documents(
    documents = split_doc1 + split_doc2,
    embedding = embeddings
)

DB_PATH = './_db/Faiss17'

db.save_local(
    folder_path=DB_PATH,
    index_name = 'faiss_index17'
)

# **피클(pickle)**은 보통 Python 객체를 파일로 저장했다가 다시 불러오는 직렬화 방식
