import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#y = wx+b 에서 w 가중치 기울기를 조정하는 걸 역전파라고 한다.
#최적의 w를 찾는 게 학습이다.

#예) x가 1일 때 y가 1이고 x가 2일 때 y가 2인걸 아는 건 역전파 덕분(기울기 조정)

#x가 고양이가 물고기를 보면
#y가 고양이가 침흘리는 거
#중간에 굉장히 많은 뉴런(신경망)이 있고, 그 신경망을 연결하는 선만큼 y = ax+b가 있음
#(그것도 순차적으로 신경망들이 이어져있다. 눈 - o layer1 - o - o - o layer4-입)
# 이렇게 신경망으로 이루어진 형태를 딥러닝(깊이있는 학습)이라고 한다.
# ai 안에 ml(머신러닝) 있고 ml(머신러닝) 안에 deep learning이 있다.
# llm은 딥러닝의 한 종류이다. 딥러닝은 신경망이 깊다. 신경망이 깊으면 학습을 잘한다.
# 왜냐하면 llm은 transformer라는 deep learning 모델을 사용하기 때문이다.

# 최소의 오차, 최적의 weigtht

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential() #신경망을 순차적으로 만든 시퀀셜모델
model.add(Dense(1, input_dim=1)) 
# input_dim=1은 x가 1차원이라는 뜻 Dense()에서 앞의 1은 뭐고 뒤의 1은 뭐지?
# 하나는 x 하나는 y라고 하셨는데

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam') 
# mse는 mean squared error(평균 제곱 오차)로 오차율을 줄이는 방법 중 하나이다.(mse는 나중에 설명해준다고 하심)
# y=ax+b의 선과 데이터들의 거리 간격 차이가 loss(손실율)
# optimizer는 손실율을 줄이는 방법을 말한다. adam은 최적화 알고리즘 중 하나이다.(adam은 그냥 외우라고 하셨음)

model.fit(x, y, epochs=3000)
#fit은 훈련시키다. (정확히는 y=ax+b 선을 30번이나 그려서 오차를 줄이는 과정을 반복)
#너무 훈련 많이 시켜도 과적합이 되어서 안좋다.

#4. 평가 예측
result = model.predict(np.array([4]))

print("4의 예측값 : ", result)