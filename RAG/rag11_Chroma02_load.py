import os
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

# #01. 데이터 불러온다.
# path = 'c:/study/furiosa-tensorflow-keras/keras/_data/rag_data/'
# loader1 = TextLoader(path + "samsung_outlook.txt", encoding='utf-8')
# loader2 = TextLoader(path + "nvidia_outlook.txt", encoding='utf-8')

# #02. 데이터 자른다.
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,  # 300개씩 자르자.
#     chunk_overlap=100,  # 중복 100개 포함.
#     separators=["\n\n", "\n", " ", ""]  # 통상 디폴트
# )
# 수업 필기: chunk_size=300이면 300개씩 자른다.
# 보완: 기본 length_function이 len이므로 여기서 300은 Token 300개가 아니라 문자열 길이 300을 뜻한다.

# split_doc1 = loader1.load_and_split(text_splitter)  # 청크 300, 오버랩 100
# split_doc2 = loader2.load_and_split(text_splitter)  # 청크 300, 오버랩 100

# #문서 개수 확인
# print(split_doc1)
# print(len(split_doc1), len(split_doc2))

# 9 -> 1536개짜리 배열이 9개 있는 것
# 수업 필기: 9 -> 1536개짜리 배열이 9개 있는 것
# 보완: split_doc1 단계에서는 아직 1536차원 벡터 9개가 아니라 Document Chunk 9개이다.
# 이후 Embedding 과정에서 각 Chunk가 1536차원 벡터 하나로 변환된다.

# [, , , , , ] 이런 요소가 1536개짜리 배열이 9개 있는 것
# 청크 한개를 나타내는 게 1536개짜리 배열로 나타내는 것.
# 즉 청크 한개가 벡터가 된다. (벡터화)
# 벡터 스토어에 넣는다.
# 수업 필기: 청크 한 개가 1536개 숫자로 이루어진 벡터 하나가 되어 Vector Store에 들어간다.
# 보완: text-embedding-3-small의 기본 출력 차원은 1536이다.
# 보완: Vector Store에는 임베딩 벡터뿐 아니라 원본 텍스트(page_content), metadata 등도 함께 저장된다.

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5
)

# 03. 벡터 스토어에 저장된 DB 불러오기
DB_PATH = './furiosa-tensorflow-keras/_db/Chroma11/'

db = Chroma(  # 불러올 때는 Chroma만 쓰면 된다.
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="croma11"
)
# 수업 필기: 저장할 때는 Chroma.from_documents(), 불러올 때는 Chroma()를 사용한다.
# 보완: 기존 persist_directory와 collection_name을 지정하면 저장된 Collection에 다시 연결하여 사용할 수 있다.
# 보완: 검색할 질문도 벡터로 바꿔야 하므로 embedding_function을 같이 지정한다.

# 저장된 데이터 확인
print("=============================================")
# print(db.get())

aaa = db.similarity_search(
    "삼성전자 사업전망에 대해 알려줘",
    k=2
)

# k = 2 는 유사한 벡터 2개를 탐색함
# 수업 필기: k=2는 유사한 벡터 2개를 탐색함
# 보완: 질문을 임베딩한 뒤 가장 가까운 Document 2개를 반환한다.
# similarity_search()의 반환값은 벡터가 아니라 list[Document]이다.

# 디폴트는 k = 4
# 수업 필기: 디폴트는 k=4
# 보완: 현재 langchain_chroma 구현에서 DEFAULT_K=4이다.

# 코사인 유사도 사용
# 수업 필기: 코사인 유사도 사용
# 보완: Chroma가 무조건 코사인 유사도를 사용하는 것은 아니다.
# 기본 거리 방식은 L2(Euclidean Distance)이고, cosine을 사용하려면 Collection 설정에서 별도로 지정해야 한다.

# 코사인 유사도는 거리가 아니라 각도로 판단한다.
# 수업 필기: 코사인 유사도는 거리가 아니라 각도로 판단한다.
# 보완: 코사인 유사도는 두 벡터의 크기보다 방향이 얼마나 비슷한지를 비교한다.
# 방향이 비슷할수록 유사도가 높고, 방향 차이가 커질수록 낮아진다.

print(aaa)
