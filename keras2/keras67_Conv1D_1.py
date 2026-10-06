# 54_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Conv1D, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [4,5,6],
    [5,6,7],
    [6,7,8],
    [7,8,9],
    [8,9,10],
])

y = np.array([4,5,6,7,8,9,10,11])

print(x.shape, y.shape)  # (8,3) (8,)
# 수업 필기: (7,3) (7,)
# 보완: 현재 x는 8행이므로 x.shape=(8,3), y.shape=(8,)이 맞다.

x = x.reshape(x.shape[0], x.shape[1], 1)  # (8,3,1)
# RNN은 input을 3차원 데이터로 만들어야 한다.
# 수업 필기: RNN은 input을 3차원 데이터로 만들어야 한다.
# 보완: RNN과 Conv1D 모두 일반적으로 (batch, timestep, feature)의 3차원 입력을 사용한다.
# 여기서는 (8,3,1) = 샘플 8개, timestep 3개, feature 1개이다.

print(x.shape)

#2. 모델구성
model = Sequential()

# model.add(SimpleRNN(units=10, input_shape=(x.shape[1],x.shape[2]), activation='linear'))
# 위의 코드는 아래의 코드와 같음
# model.add(SimpleRNN(10, input_shape=(x.shape[1],x.shape[2])))
# 수업 필기: 위의 두 SimpleRNN 코드는 같다.
# 보완: units=10을 쓰는 방식은 같지만 activation이 다르므로 완전히 같은 코드는 아니다.
# 첫 번째는 activation='linear', 두 번째는 기본 activation='tanh'가 적용된다.

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
# Conv1D 입력 shape = (batch, timestep, feature)
# 실제 입력 shape = (8,3,1)
# filters=10 → 출력 channel(feature) 개수 10개
# kernel_size=2 → 시간축에서 연속된 timestep 2개씩 보면서 특징을 추출
# padding 기본값은 'valid'이므로 timestep 길이가 줄어든다.
# 입력 timestep 3 → 출력 timestep 2
# 출력 shape = (8,2,10)

# conv2D면 kernel_size가 (2,2) 인데, conv1D면 (2,) 로 표현한다.
# 수업 필기: Conv2D는 kernel_size=(2,2), Conv1D는 (2,)로 표현한다.
# 보완: Conv1D에서는 kernel_size=2처럼 정수 하나로 적는 것이 일반적이다.

model.add(Conv1D(10,2))
# Conv1D(10,2) = filters=10, kernel_size=2
# 입력 shape = (8,2,10)
# padding='valid'이므로 timestep 2 → 1
# 출력 shape = (8,1,10)

model.add(Flatten())
# 입력 shape = (8,1,10)
# 출력 shape = (8,10)

model.add(Dense(1, activation='linear'))
# 최종 출력 shape = (8,1)

# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
# 수업 필기: 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
# 보완: Conv1D의 출력은 3차원이고, Flatten이 이를 2차원으로 펴준 뒤 Dense에 연결한다.
# Dense(1)의 최종 출력은 (batch, 1)이다.

# Shape 흐름
# (8,3,1)
#   ↓ Conv1D(10,2)
# (8,2,10)
#   ↓ Conv1D(10,2)
# (8,1,10)
#   ↓ Flatten()
# (8,10)
#   ↓ Dense(1)
# (8,1)

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os, datetime

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

filepath = './_save/keras67/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

date = datetime.datetime.now().strftime("%m%d_%H%M")

# 1. 모델을 저장하는 콜백 (ModelCheckpoint)
mcp = ModelCheckpoint(
    monitor='loss',
    mode='min',
    verbose=0,
    save_best_only=True,
    filepath=filepath + 'k67_' + date + '_{epoch:04d}-{loss:.4f}.keras'
)
# monitor='loss' → 훈련 loss 감시
# mode='min' → loss는 작을수록 좋음
# save_best_only=True → 이전 최고 성능보다 좋아졌을 때만 저장
# 수업 필기에서 MCP라고 줄여 적을 수 있지만 정확한 클래스명은 ModelCheckpoint이다.

# 2. 학습률을 조금씩 줄여주는 콜백 (ReduceLROnPlateau)
rlr = ReduceLROnPlateau(
    monitor='loss',   # 개선되는지 지켜볼 지표 (여기선 검증 데이터가 없으므로 loss 사용)
    patience=50,      # 50번의 에포크 동안 loss가 안 떨어지면
    mode='min',
    verbose=1,
    factor=0.5        # 학습률을 절반(0.5)으로 줄여서 더 세밀하게 탐색함
)
# 수업 필기: loss가 50 Epoch 동안 개선되지 않으면 learning rate를 절반으로 줄인다.
# 보완: 정확히는 현재 learning_rate × factor이므로 factor=0.5이면 절반이 된다.

model.fit(
    x,
    y,
    epochs=1000,
    batch_size=14,
    callbacks=[mcp, rlr]
)

# 수업 필기: batch_size=14
# 보완: 현재 전체 샘플이 8개뿐이므로 batch_size=14라고 해도 실제 한 Epoch에서는 8개 전체가 한 Batch로 처리된다.
# 데이터 개수보다 batch_size가 크다고 오류가 발생하는 것은 아니다.

#4. 평가 예측
results = model.evaluate(x,y)
print("loss :", results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
# 예측 데이터도 Conv1D 입력 shape에 맞춰 (batch, timestep, feature) 형태로 만든다.
# (1,3,1) = 샘플 1개, timestep 3개, feature 1개

y_predict = model.predict(x_predict)
print('[8,9,10]의 결과 : ', y_predict)

# 학습 패턴
# [1,2,3] → 4
# [2,3,4] → 5
# ...
# [8,9,10] → 11
# 충분히 학습되었다면 [8,9,10] 입력에 대해 11에 가까운 값을 기대한다.