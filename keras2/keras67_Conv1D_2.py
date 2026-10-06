# 55_2 카피

from tensorflow.keras.layers import Dropout
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x = np.array([
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [4,5,6],
    [5,6,7],
    [6,7,8],
    [7,8,9],
    [8,9,10],
    [9,10,11],
    [10,11,12],
    [20,30,40],
    [30,40,50],
    [40,50,60]
])

#원데이터는 1,2,3,4,5,6,7,8,9,10,20,30,40
# 수업 필기: 원데이터는 1,2,3,4,5,6,7,8,9,10,20,30,40
# 보완: 현재 x는 단순 원데이터 자체가 아니라 3개 timestep씩 묶은 입력 샘플 13개이다.
#       x.shape = (13,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# 주가가 급상승한 데이터라고 생각하자.
# RNN은 뒤쪽의 영향을 많이 받는다.
# 수업 필기: RNN은 뒤쪽의 영향을 많이 받는다.
# 보완: 기본 RNN은 timestep을 순서대로 처리하고 마지막 hidden state에 앞의 정보를 누적한다.
#       그래서 가까운 최근 timestep의 영향이 강해질 수 있지만 무조건 뒤쪽 데이터만 중요하다는 뜻은 아니다.
#       LSTM/GRU는 장기 정보를 더 잘 보존하도록 Gate 구조를 사용한다.

x_predict = np.array([[50,60,70]])  # 80을 맞춰보아요.
y_predict = np.array([80])

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

# x.shape = (13,3,1)
# x_predict.shape = (1,3,1)
# (batch, timestep, feature)
# x = 샘플 13개, timestep 3개, feature 1개
# x_predict = 샘플 1개, timestep 3개, feature 1개

#2. 모델구성
model = Sequential()

# model.add(SimpleRNN(units=5, input_shape=(x.shape[1],x.shape[2]), activation='linear'))
# model.add(LSTM(10, input_shape=(3,1), activation='linear'))
# model.add(GRU(10, input_shape=(3,1)))
# 수업 필기: 같은 3차원 시계열 데이터를 SimpleRNN, LSTM, GRU, Conv1D 등에 입력할 수 있다.
# 보완: 입력 shape는 비슷하지만 각 Layer가 시계열 정보를 처리하는 방식은 서로 다르다.

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
# 입력 shape = (13,3,1)
# filters=10 → 출력 feature(channel) 10개
# kernel_size=2 → 시간축에서 연속된 timestep 2개씩 본다.
# padding 기본값='valid'이므로 timestep 3 → 2
# 출력 shape = (13,2,10)

model.add(Conv1D(10, 2))
# Conv1D(10,2) = filters=10, kernel_size=2
# 입력 shape = (13,2,10)
# timestep 2 → 1
# 출력 shape = (13,1,10)

model.add(Flatten())
# 입력 shape = (13,1,10)
# 출력 shape = (13,10)

# 위의 코드는 아래의 코드와 같음
# model.add(SimpleRNN(10, input_shape=(x.shape[1],x.shape[2])))
# 수업 필기: 위의 코드는 아래의 SimpleRNN 코드와 같음
# 보완: Conv1D + Conv1D + Flatten과 SimpleRNN은 같은 코드가 아니다.
#       최종적으로 비슷한 형태의 2차원 출력을 만들 수는 있지만 내부 연산 방식은 완전히 다르다.
#       SimpleRNN은 hidden state를 timestep마다 전달하고, Conv1D는 이동하는 kernel로 국소 패턴을 추출한다.

model.add(Dense(3, activation='relu'))
# 입력 shape = (13,10)
# 출력 shape = (13,3)

model.add(Dense(1))
# 출력 shape = (13,1)

# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
# 수업 필기: 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
# 보완: Conv1D 출력은 3차원이지만 Flatten()으로 2차원 (batch, feature) 형태로 바꾼 뒤 Dense에 연결한다.

# Shape 흐름
# (13,3,1)
#    ↓ Conv1D(10,2)
# (13,2,10)
#    ↓ Conv1D(10,2)
# (13,1,10)
#    ↓ Flatten()
# (13,10)
#    ↓ Dense(3)
# (13,3)
#    ↓ Dense(1)
# (13,1)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=2000,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=2000,
    verbose=1,
    factor=0.5,
)

# 수업 필기: EarlyStopping과 ReduceLROnPlateau에서 val_loss를 감시한다.
# 보완: 현재 model.fit()에 validation_data 또는 validation_split이 없기 때문에 val_loss가 생성되지 않는다.
#       따라서 현재 코드 그대로라면 val_loss를 제대로 감시할 수 없다.
#
# 방법 1: validation_split 사용
# model.fit(x, y, validation_split=0.2, ...)
#
# 방법 2: validation_data=(x_val, y_val) 사용
#
# 검증 데이터를 사용하지 않을 것이라면 monitor='loss'로 바꿔야 한다.

# patience=2000
# 수업 필기: 2000 Epoch 동안 개선되지 않을 경우 EarlyStopping / Learning Rate 감소
# 보완: 현재 epochs=1000인데 patience=2000이므로 훈련 도중 patience 조건에 도달할 수 없다.
#       즉 사실상 EarlyStopping과 ReduceLROnPlateau가 작동하지 않는다.

# path = './_save/keras34/'
# import os
# os.makedirs(path, exist_ok=True)

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
#     save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
#     filepath=path + 'keras55_LSTM2_scale.keras',
#     verbose=1,
# )
# 수업 필기: val_loss는 낮을수록 좋으므로 mode='auto'에서 min 방향으로 판단한다.
# 보완: 명확하게 mode='min'으로 직접 지정해도 된다.
# 보완: 이 ModelCheckpoint 역시 val_loss를 사용하려면 Validation 데이터가 있어야 한다.

model.compile(loss='mse', optimizer='adam')

model.fit(
    x,
    y,
    epochs=1000,
    batch_size=13,
    callbacks=[es, rlr]
)

# 수업 필기: batch_size=13
# 보완: 현재 전체 학습 Sample이 정확히 13개이므로 한 Batch에 전체 13개 데이터가 들어간다.
#       따라서 1 Epoch당 기본적으로 Gradient Update가 1번 발생한다.

#4. 평가 예측
print(model.evaluate(x_predict, y_predict))
# x_predict = (1,3,1)
# y_predict = (1,)
# 실제 정답 80과 모델 예측값 사이의 MSE를 계산한다.

print(" 예측값 : ", model.predict(x_predict))
# 학습 데이터 후반 패턴:
# [20,30,40] → 50
# [30,40,50] → 60
# [40,50,60] → 70
#
# 예측:
# [50,60,70] → 80 기대
#
# 보완: 80은 학습 범위 밖의 값을 예측하는 외삽(extrapolation)이므로
#       모델이 반드시 정확하게 80을 출력한다고 보장할 수는 없다.