# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

# parser 분석하다. 답변을 다듬다.
# 수업 필기: parser 분석하다. 답변을 다듬다.
# 보완: Output Parser는 LLM의 출력 결과를 원하는 형태(str, JSON, 객체 등)로 변환/가공한다.

# 페르소나: 역할을 주는 것
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화를 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

# 상황:
{question}

# FORMAT:
- 영어회화 :
- 한글번역 :
"""

# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MONOROUTER_API_KEY").strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template(template=template)
print(prompt)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5
)

# open ai 디멘션 디폴트가 1536
# 요소가 1536개짜리 배열로 변환시킨다.
# numpy
# 수업 필기: open ai 디멘션 디폴트가 1536, 요소가 1536개짜리 배열로 변환시킨다.
# 보완: text-embedding-3-small의 기본 임베딩 차원은 1536이다.
# 즉, 문장 하나를 입력하면 [0.012, -0.034, ...]처럼 숫자 1536개로 이루어진 벡터가 생성된다.
# dimensions 값을 지정하면 지원 범위 내에서 출력 벡터의 차원을 줄여 사용할 수 있다.
# embed_query()의 반환값은 일반적으로 Python list[float]이며, 필요하면 np.array(vector)로 NumPy 배열로 변환할 수 있다.

# embed_query는 문자열(str)을 입력받아야 합니다.
# PromptTemplate 객체(prompt)를 그대로 넣으면 에러가 발생하므로 문자열을 넣습니다.
text = "저는 부산에서 밀면을 먹고싶어요."
vector = embeddings.embed_query(text)

print(vector)
print("=" * 10)
print("임베딩 벡터의 차원 :", len(vector))

DB_PATH = "./_db/Chroma/"
# Chroma Vector DB를 로컬에 저장할 경로