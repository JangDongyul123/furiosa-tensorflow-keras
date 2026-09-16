# 36 - Stacking Ensemble (CIFAR-10) - Recompile Test

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =================================================================================
# 1. 데이터
# =================================================================================
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 스케일링 (Max ABS -> -1.0 ~ 1.0)
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

# OneHotEncoding
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# =================================================================================
# 2. 모델 구성
# =================================================================================

model = Sequential()


# ---------------------------------------------------------------------------------
# Conv2D 1
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    512,
    kernel_size=(3, 3),
    input_shape=(32, 32, 3),
    padding='same'
))
model.add(MaxPooling2D())

# 입력 shape : (32, 32, 3)
#
# 32 : 이미지의 세로 크기
# 32 : 이미지의 가로 크기
# 3  : 채널(channel)의 개수
#
# CIFAR-100은 RGB 컬러 이미지이므로 채널이 3개이다. (32, 32, 3)
#
# --------------------------------------------------------------------
# Conv2D(512, kernel_size=(3,3), padding='same') & MaxPooling2D()
# --------------------------------------------------------------------
#
# 512는 "필터의 개수"이다.
#
# kernel_size=(3,3)은 필터의 가로/세로 크기만 지정한다.
# 실제 필터 하나의 크기는 입력 채널의 깊이까지 포함해서 (3, 3, 3) 이다.
#
# 즉,
# 입력           : (32, 32, 3)
# 필터 하나      : (3, 3, 3)
# 필터 개수      : 512개
#
# padding='same'이므로 Conv2D를 통과해도 가로/세로 크기는 (32, 32)로 유지되며,
# 필터가 512개 있으므로 채널이 512가 된다. -> (32, 32, 512)
#
# 이후 MaxPooling2D()를 통과하면 가로/세로 크기가 절반으로 줄어든다.
#
# 따라서 최종 출력 shape:
#     (16, 16, 512)


# ---------------------------------------------------------------------------------
# Conv2D 2
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    256,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# 이전 층의 출력:
#
#     (16, 16, 512)
#
# 따라서 현재 Conv2D가 받는 입력의 채널 깊이는 512이다.
#
# kernel_size=(3,3)이라고 적었지만 실제 필터 하나의 크기는 (3, 3, 512) 이다.
# 여기서 512는 "입력 채널의 깊이"이다.
#
# 한 위치에서 3 x 3 x 512 = 4,608개의 값을 계산한다.
#
# 필터 하나   : (3, 3, 512)
# 필터 개수   : 256개
#
# padding='same'이므로 가로/세로 크기는 유지된다.
#
# 출력 shape:
#
#     (16, 16, 256)


# ---------------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------------

# model.add(Dropout(0.2))

# 학습할 때 뉴런(출력값)의 20%를 랜덤하게 0으로 만들어
# 특정 뉴런에 지나치게 의존하는 것을 줄인다.
#
# Dropout은 데이터의 shape를 변경하지 않는다.
#
# 입력: (16, 16, 256)
# 출력: (16, 16, 256)


# ---------------------------------------------------------------------------------
# Conv2D 3
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    128,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# 입력:
#
#     (16, 16, 256)
#
# 입력 채널이 256이므로 필터 하나의 실제 크기는 (3, 3, 256)이다.
# 이런 필터가 128개 있다.
#
# padding='same'이므로 크기 유지.
#
# 출력 shape:
#
#     (16, 16, 128)


# ---------------------------------------------------------------------------------
# Conv2D 4
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# 입력:
#
#     (16, 16, 128)
#
# 필터 개수: 64개
#
# 출력 shape:
#
#     (16, 16, 64)


# ---------------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------------

# model.add(Dropout(0.2))

# shape는 그대로 유지된다.
#
# 입력: (16, 16, 64)
# 출력: (16, 16, 64)


# ---------------------------------------------------------------------------------
# Conv2D 5
# ---------------------------------------------------------------------------------

model.add(Conv2D(
    8,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# 입력:
#
#     (16, 16, 64)
#
# 필터 개수: 8개
#
# 출력 shape:
#
#     (16, 16, 8)


model.add(Flatten())

# ---------------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------------
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(
    100,
    activation='softmax'
))

# 주의:
# 
# 이전 Conv2D의 출력은 (16, 16, 8) 이었다.
# 이를 Flatten() 층에 통과시키면 1차원 배열로 평탄화된다.
# 
# 16 x 16 x 8 = 2048
# 즉, Flatten() 층의 출력 shape는 (2048,) 이 된다.
# 
# 그 후 Dense 층들을 거치며 최종적으로 100개의 클래스에 대한 확률을 출력한다.
# 최종 출력 shape: (100,)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs = 30, batch_size = 100, verbose=1)
model.fit(x_train, y_train, epochs = 100, batch_size = 2000, verbose=1)

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
# (10000, 100)

# argmax는 가장 큰 값의 인덱스를 반환한다.
# 즉, 가장 확률이 높은 클래스의 인덱스를 반환한다.
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc_score)
print("걸린 시간: ", end - start)
