from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4 ,3, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(2, input_dim=1))
model.add(Dense(1, input_dim=2))
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y,epochs=500)

#4. 평가 예측
loss = model.evaluate(x,y)
print("loss: "+str(loss))

result = model.predict(np.array([1,2,3,4,5,6]))
print("예측값: "+str(result))