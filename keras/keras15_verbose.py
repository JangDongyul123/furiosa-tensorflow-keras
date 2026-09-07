# [실습] verbose 옵션 이해하기

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
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose = 1) 
"""
[ verbose 매개변수 ]
- 0: 훈련 과정 출력 안 함 (침묵 모드)
- 1: 디폴트 출력 (프로그레스 바 및 훈련 과정 상세 출력)
- 2: 프로그레스 바 제외하고 출력
- 3 이상: 에포크(epoch) 횟수만 출력

* 로그를 화면에 출력하는 작업은 리소스(시간, 성능)를 소모합니다. 
* verbose를 적절히 조절하여 학습 속도를 높이고 불필요한 출력을 줄일 수 있습니다.
"""

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 



