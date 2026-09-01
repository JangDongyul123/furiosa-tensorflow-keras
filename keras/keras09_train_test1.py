import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10) 
# 훈련데이터가 7개 이상이면 epochs 1번 당 통훈련 1번이므로 w갱신 1번
# batch_size가 4개면 epochs 1번 당 훈련 2번이므로 w갱신 2번
# 그리고 batch_size가 1개면 신경망이 1개고, batch_size가 2개면 병렬로 계산해야해서 신경망이 2개가 되는 줄 알았는데
# 그게 아니라 batch_size가 1개일 때 scala 데이터 1개라면
# batch_size가 2개면 벡터 데이터 1개로 전달된다고 하심

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
#평가를 테스트 셋으로 하겠다. 훈련을 안해본 데이터이기 때문에 과적합걸리지 않음
#즉, loss = model.evaluate(x_train, y_train) 보다 
# loss = model.evaluate(x_test, y_test) 가 신뢰가 가능하다
# 통상적으로 평가 loss가 학습 loss보다 나쁘다.


