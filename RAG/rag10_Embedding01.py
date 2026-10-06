# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

# parser 분석하다. 답변을 다듬다.
# 수업 필기: parser 분석하다. 답변을 다듬다.
# 보완: Output Parser는 LLM의 출력 결과를 원하는 Python 형태(str, JSON, 객체 등)로 변환/가공한다.

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

prompt_template = "{topic}에 대해 간단히 설명해 줘."
prompt = PromptTemplate.from_template(template=template)
print(prompt)

#
# model = ChatOpenAI(
#     model='gpt-5.6-terra',
#     # openai_api_key=openai_api_key,
#     api_key=api_key,
#     temperature=0,
#     base_url=base_url,
# )

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)

# embed_query는 문자열(str)을 입력받아야 합니다.
# PromptTemplate 객체(prompt)를 그대로 넣으면 에러가 발생하므로 문자열을 넣습니다.
text = "저는 부산에서 밀면을 먹고싶어요."
vector = embeddings.embed_query(text)

print(vector)
print("=" * 10)
print("임베딩 벡터의 차원 :", len(vector))

# AI는 입력을 받으면 수치로 인식하고, 수치화된 OUTPUT 연산의 결과를 String형태로 반환한다.
# 수업 필기: AI는 입력을 받으면 수치로 인식하고, 수치화된 OUTPUT 연산의 결과를 String형태로 반환한다.
# 보완: LLM 내부에서는 입력 토큰을 임베딩 벡터 등의 수치 표현으로 처리하지만,
#       모델 자체의 반환값은 보통 AIMessage 객체이고, 그 안의 content에 문자열 답변이 들어간다.

# String 형태로 반환하게 해주는 게 StrOutParser
# 수업 필기: String 형태로 반환하게 해주는 게 StrOutParser
# 보완: 정확한 클래스명은 StrOutputParser이며, AIMessage의 content를 꺼내 문자열(str)로 반환해준다.
# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser
# 보완: | 연산자로 Runnable들을 연결하여 하나의 실행 파이프라인(chain)을 구성한다.

# input = {"question": "저는 부산에서 밀면을 먹고싶어요."}

# invoke는 model.predict와 같다.
# 수업 필기: invoke는 model.predict와 같다.
# 보완: 둘 다 "입력을 넣어 한 번 실행한다"는 의미로 이해하면 되지만,
#       invoke는 LangChain Runnable의 공통 실행 인터페이스이고 predict와 완전히 같은 개념은 아니다.
# response = chain.invoke(input)

# print(response.content) out_parser 덕분에 .content 를 안써도 String 형태로 출력된다.
# 수업 필기: out_parser 덕분에 .content 를 안써도 String 형태로 출력된다.
# 보완: StrOutputParser가 AIMessage.content를 추출해 str로 반환하므로 response.content가 아니라 response를 바로 출력한다.
# print(response)
