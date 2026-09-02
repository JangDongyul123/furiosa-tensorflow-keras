import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

#1. 데이터 구성
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, test_size=0.3)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(13, input_dim = 13))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size =100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x)
# print("results: ", results)