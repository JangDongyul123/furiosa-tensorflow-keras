import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x=x.transpose()

# 2. 모델구성
model = Sequential()
model.add(Dense(3,input_dim=3))
model.add(Dense(1)) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1, batch_size=2)

# 4. 평가 예측
loss = model.evaluate(x,y)
print('loss: ', loss) #0.0015
results = model.predict(np.array([[10,1.3,0]]));  #10.00
print('results: ', results)
