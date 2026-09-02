from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, test_size=0.3,random_state=333)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 100)

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
results = model.predict(x , batch_size = 32)