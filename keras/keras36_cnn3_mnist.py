# 36 - 2번 카피

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

x_train = x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28,28, 1)

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) #디폴트 값은 True
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train) # 2차원을 받는다.
y_test = ohe.transform(y_test)

print(y_train.shape, y_test.shape)
# (60000, 10), (10000, 10)



# =================================================================================
# 2. 모델 구성
# =================================================================================

model = Sequential()


# ---------------------------------------------------------------------------------
# Conv2D 1
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    input_shape=(28, 28, 1)
))
model.add(Dropout(0.2))

# 입력 shape : (28, 28, 1)
#
# 28 : 이미지의 세로 크기
# 28 : 이미지의 가로 크기
# 1  : 채널(channel)의 개수
#
# 흑백 이미지는 채널이 1개이므로 (28, 28, 1)
# RGB 컬러 이미지라면 채널이 3개이므로 (28, 28, 3)
#
#
# --------------------------------------------------------------------
# Conv2D(64, kernel_size=(3,3))
# --------------------------------------------------------------------
#
# 64는 "필터의 개수"이다.
# kernel_size=(3,3)은 필터의 가로/세로 크기만 지정한다.
# 실제 필터 하나의 크기는 입력 채널의 깊이까지 포함해서 (3, 3, 1) 이다.
#
# 입력           : (28, 28, 1)
# 필터 하나      : (3, 3, 1)
# 필터 개수      : 64개
#
# padding='valid'가 기본값이고 stride=1이므로
# 출력 크기 공식: (28 - 3) / 1 + 1 = 26
#
# 따라서 출력 shape:
#     (26, 26, 64)


# ---------------------------------------------------------------------------------
# Conv2D 2
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='relu'
))

# 이전 층의 출력:
#     (26, 26, 64)
#
# 따라서 현재 Conv2D가 받는 입력의 채널 깊이는 64이다.
#
# 실제 필터 하나의 크기는 (3, 3, 64) 이다.
# 필터 하나   : (3, 3, 64)
# 필터 개수   : 32개
#
# 가로/세로: (26 - 3) / 1 + 1 = 24
#
# 출력 shape:
#     (24, 24, 32)


# ---------------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------------

# model.add(Dropout(0.2))

# 학습할 때 뉴런(출력값)의 20%를 랜덤하게 0으로 만들어
# 특정 뉴런에 지나치게 의존하는 것을 줄인다.
#
# Dropout은 데이터의 shape를 변경하지 않는다.
#
# 입력  : (26, 26, 64)
# 출력  : (26, 26, 64)


# ---------------------------------------------------------------------------------
# Conv2D 3
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='relu'
))

# 입력: (24, 24, 32)
# 필터 하나의 실제 크기: (3, 3, 32)
# 필터 개수: 32개
#
# 가로/세로: (24 - 3) + 1 = 22
#
# 출력 shape:
#     (22, 22, 32)


# ---------------------------------------------------------------------------------
# Conv2D 4
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    16,
    kernel_size=(3, 3),
    activation='relu'
))

# 입력: (22, 22, 32)
# 필터 하나의 실제 크기: (3, 3, 32)
# 필터 개수: 16개
#
# 가로/세로: (22 - 3) + 1 = 20
#
# 출력 shape:
#     (20, 20, 16)


# ---------------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------------

# model.add(Dropout(0.2))

# shape는 그대로 유지된다.
#
# 입력  : (20, 20, 16)
# 출력  : (20, 20, 16)


# ---------------------------------------------------------------------------------
# Conv2D 5
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    8,
    kernel_size=(3, 3),
    activation='relu'
))

# 입력: (20, 20, 16)
# 필터 하나의 실제 크기: (3, 3, 16)
# 필터 개수: 8개
#
# 가로/세로: (20 - 3) + 1 = 18
#
# 출력 shape:
#     (18, 18, 8)


model.add(Flatten())

# ---------------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------------

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, input_shape=(32, ), activation='relu'))

model.add(Dense(
    10,
    activation='softmax'
))

# 주의:
#
# 이전 Conv2D의 출력은 (18, 18, 8) 이었다.
# 이를 Flatten() 층에 통과시키면 1차원 배열로 평탄화된다.
#
# 18 x 18 x 8 = 2592
# 즉, Flatten() 층의 출력 shape는 (2592,) 이 된다.
#
# 그 후 Dense 층들을 거치며 최종적으로 10개의 클래스에 대한 확률을 출력한다.
# 최종 출력 shape: (10,)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs = 30, batch_size = 100, verbose=1)
model.fit(x_train, y_train, epochs = 50, batch_size = 2000, verbose=1)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 30, batch_size = 100, verbose=1)
model.fit(x_train, y_train, epochs = 100, batch_size = 2000, verbose=1)

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