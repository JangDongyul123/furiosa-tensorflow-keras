import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).transpose()

print(x.shape, y.shape)

# 실습
# 11, 0, -1 이 나오면 땡큐

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(6))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x,y,epochs=1000, batch_size=32)

#4. 평가 예측
loss = model.evaluate(x,y)
print("loss: ",loss)

result = model.predict(np.array([[10]]))
print("result: ",result)

#loss가 높은데, 예측결과가 좋은 것보다
#예측 결과가 나빠도 loss가 낮은 게 더 좋다