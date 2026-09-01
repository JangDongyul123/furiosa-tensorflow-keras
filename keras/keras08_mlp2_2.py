import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # 0부터 10-1까지 즉, 0,1,2,3,4,5,6,7,8,9
print(x)

x = np.array(range(1,10))
print(x) # 1,2,3,4,5,6,7,8,10

x = np.array([range(10),range(21,31,), range(201,211)]).T
print(x.shape) #(3,10) ->(10,3)

y = np.array(range(1,11))
print(y.shape) #(10,)

#2. 모델 구성
model= Sequential()
model.add(Dense(3,input_dim = 3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=32)

# 4. 평가 예측
loss = model.evaluate(x,y)
print('loss: ', loss)
result = model.predict(np.array([[10,31,211]]))
print('result: ', result)

#[10,31,211]
#11.00까지 나오면 합격