import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

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

# rag17_FAISS_1_save.py에서 저장한 인덱스만 읽습니다.
DB_PATH = "./_db/Faiss17"
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print(f"FAISS에 저장된 문서 수: {db.index.ntotal}")
result = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(result)
