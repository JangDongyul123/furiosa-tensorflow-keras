from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) #하이퍼파라미터 튜닝
model.add(Dense(2, input_dim=1))
model.add(Dense(1, input_dim=2))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y,epochs=1000) #batch_size=3 이면 데이터 3개씩, 디폴트는 32개씩 학습한다.

#4. 평가 예측
loss = model.evaluate(x,y)
print("loss: "+str(loss))

# result = model.predict(np.array([1,2,3,4,5,6]))
# print("예측값: "+str(result))