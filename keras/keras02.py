from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np


# 1. 데이터
# x = 입력 데이터
# y = 정답 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])


# 2. 모델 구성
model = Sequential()

# Dense Layer 1개를 추가한다.
# Dense(1)의 1 = 출력 뉴런 개수 1개
# input_dim=1 = 한 샘플의 입력 feature 개수 1개
model.add(Dense(1, input_dim=1))


# 3. 컴파일 및 훈련

# loss='mse'
# → 예측값과 정답의 오차를 MSE(평균 제곱 오차)로 계산한다.
#
# optimizer='adam'
# → 역전파로 계산한 gradient를 이용해
#   weight와 bias를 수정하여 loss를 줄이는 optimizer로 Adam을 사용한다.
model.compile(loss='mse', optimizer='adam')


# 전체 학습 데이터 x, y를 100번 반복해서 학습한다.
model.fit(x, y, epochs=100)


# 4. 평가

# 현재 모델로 x를 예측한 뒤,
# 실제 정답 y와 비교하여 MSE loss를 계산한다.
loss = model.evaluate(x, y)

print('loss : ', loss)


# 5. 예측

# 학습된 모델에 새로운 x값들을 넣어서 y를 예측한다.
result = model.predict(np.array([1, 2, 3, 4, 5, 6, 7]))

print("예측값 : ", result)


# loss는 일반적으로 0에 가까울수록 좋다.
# loss = 0이면 현재 평가한 데이터에 대해서는
# 예측값과 정답이 완전히 일치했다는 뜻이다.
#
# 하지만 훈련 데이터의 loss가 0이라고 해서
# 새로운 데이터까지 잘 맞추는 완벽한 모델이라는 뜻은 아니다.
# 과적합(overfitting)이 발생할 수도 있다.