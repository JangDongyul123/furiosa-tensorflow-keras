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
#
# kernel_size=(3,3)은 필터의 가로/세로 크기만 지정한다.
#
# 실제 필터 하나의 크기는 입력 채널의 깊이까지 포함해서
#
#     (3, 3, 1)
#
# 이다.
#
# 즉,
#
# 입력           : (28, 28, 1)
# 필터 하나      : (3, 3, 1)
# 필터 개수      : 64개
#
#
# 필터 하나는 입력의 3 x 3 x 1 영역을 계산하여
# 한 위치에서 숫자 1개를 만든다.
#
# 그리고 필터가 64개 있으므로
# feature map도 64개가 만들어진다.
#
#
# padding='valid'가 기본값이고 stride=1이므로
#
# 출력 크기 공식:
#
#     (입력 크기 - 필터 크기) / stride + 1
#
#     (28 - 3) / 1 + 1
#     = 26
#
# 따라서 출력 shape:
#
#     (26, 26, 64)
#
# 마지막 64는 "필터의 개수 = 출력 채널의 개수"이다.


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
#     (26, 26, 64)
#
# 따라서 현재 Conv2D가 받는 입력의 채널 깊이는 64이다.
#
# kernel_size=(3,3)이라고 적었지만
# 실제 필터 하나의 크기는:
#
#     (3, 3, 64)
#
# 이다.
#
# 여기서 매우 중요:
#
# (3, 3, 64)의 64는 "필터 개수"가 아니라
# "필터 하나의 깊이"이다.
#
# 입력의 깊이가 64이기 때문에
# 필터도 깊이 64까지 한꺼번에 계산한다.
#
#
# 한 위치에서:
#
#     3 x 3 x 64 = 576개 값
#
# 을 각각 가중치와 곱한 뒤 전부 더해서
# 숫자 1개를 만든다.
#
#
# 그런데 Conv2D(32, ...)이므로
# 이런 필터가 32개 존재한다.
#
# 필터 하나   : (3, 3, 64)
# 필터 개수   : 32개
#
# 필터 1개 → feature map 1개
# 필터 32개 → feature map 32개
#
#
# 가로/세로:
#
#     (26 - 3) / 1 + 1
#     = 24
#
# 출력 shape:
#
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
# 입력  : (24, 24, 32)
# 출력  : (24, 24, 32)


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
#     (24, 24, 32)
#
# 입력 채널이 32이므로
# 필터 하나의 실제 크기:
#
#     (3, 3, 32)
#
# 이런 필터가 32개 있다.
#
# 가로/세로:
#
#     (24 - 3) + 1
#     = 22
#
# 출력 shape:
#
#     (22, 22, 32)


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
#     (22, 22, 32)
#
# 필터 하나의 실제 크기:
#
#     (3, 3, 32)
#
# 필터 개수:
#
#     16개
#
# 가로/세로:
#
#     (22 - 3) + 1
#     = 20
#
# 출력 shape:
#
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
    activation='relu',
    padding='same'
))

# 입력:
#
#     (20, 20, 16)
#
# 필터 하나의 실제 크기:
#
#     (3, 3, 16)
#
# 필터 개수:
#
#     16개
#
# 가로/세로:
#
#     (20 - 3) + 1
#     = 18
#
# 출력 shape:
#
#     (18, 18, 16)
#
# ※ 기존 주석의 (20,20,16)은 잘못된 값이다.


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

# 주의!!!
#
# 현재 Conv2D의 출력은
#
#     (18, 18, 16)
#
# 이다.
#
# 여기서 바로 Dense(10)을 사용하면
#
#     (10,)
#
# 이 되는 것이 아니다.
#
# Dense는 마지막 차원 16에 대해서 적용되기 때문에
#
#     (18, 18, 16)
#           ↓
#       Dense(10)
#           ↓
#     (18, 18, 10)
#
# 이 된다.
#
# 즉 현재 모델의 최종 출력 shape는:
#
#     (18, 18, 10)
#
# 이다.

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
# (10000, 18, 18, 10)

# argmax는 가장 큰 값의 인덱스를 반환한다.
# 즉, 가장 확률이 높은 클래스의 인덱스를 반환한다.
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc_score)
print("걸린 시간: ", end - start)
