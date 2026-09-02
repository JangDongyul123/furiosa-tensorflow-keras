#09_1 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱

x_train = x[0:7] # 항상 인덱스 시작은 0부터, x_train = x[:7]
y_train = y[:7]

x_test = x[7:10]
y_test = y[7:]

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=10) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
#평가를 테스트 셋으로 하겠다. 훈련을 안해본 데이터이기 때문에 과적합걸리지 않음
#즉, loss = model.evaluate(x_train, y_train) 보다 
# loss = model.evaluate(x_test, y_test) 가 신뢰가 가능하다
# 통상적으로 평가 loss가 학습 loss보다 나쁘다.
