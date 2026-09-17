# accuracy_score 0.9816

from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
# (60000, 28, 28), (60000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (60000, 28, 28, 1)

print(x_test.shape, y_test.shape) 
# (10000, 28, 28), (10000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (10000, 28, 28, 1)

# print(x_train[0])

print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

#### 스케일링 1 - min - max 스케일링
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

#### 스케일링 2 - max ABS 스케일링
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
print(x_train.shape, x_test.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) #디폴트 값은 True
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train) # 2차원을 받는다.
y_test = ohe.transform(y_test)


# =================================================================================
# 2. 모델 구성
# =================================================================================

model = Sequential()
# ---------------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------------
model.add(Dense(units=64, input_shape=(28*28, )))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(
    10,
    activation='softmax'
))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs = 30, batch_size = 100, verbose=1)
model.fit(x_train, y_train, epochs = 500, batch_size = 200000, verbose=1)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 30, batch_size = 100, verbose=1)
model.fit(x_train, y_train, epochs = 500, batch_size = 200000, verbose=1)

end = time.time()
print(f"{end - start}")


#4. 평가, 예측
print("=============== model.evaluate ===================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_pred = model.predict(x_test)
print(y_pred.shape)
# (10000, 10)

# argmax는 가장 큰 값의 인덱스를 반환한다.
# 즉, 가장 확률이 높은 클래스의 인덱스를 반환한다.
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc_score)
print("걸린 시간: ", end - start)