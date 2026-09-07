# [실습] 조기 종료(Early Stopping) 기법 적용하기 - 캘리포니아 주택 가격 데이터셋

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, test_size=0.3,random_state=333)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8 , activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(4, activation= 'relu'))
model.add(Dense(2, activation= 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping 
# 파이썬에서는 흔하지 않지만, 자바 개발에서는 흔한 카멜케이스.

from tensorflow.keras.callbacks import EarlyStopping
import time


# ============================================================
# EarlyStopping 설정
# ============================================================

es = EarlyStopping(

    monitor='val_loss',
    # 어떤 값을 기준으로 학습 중단 여부를 판단할지 설정
    #
    # 'val_loss' :
    # validation 데이터의 loss를 감시한다.
    #
    # 즉, 각 epoch가 끝날 때마다
    # validation 데이터에 대한 loss를 확인한다.
    #
    # train loss가 아니라 val_loss를 보는 이유는
    # 훈련 데이터에만 과적합되고 있는지 확인하기 위해서이다.


    mode='min',
    # monitor로 지정한 값이
    # 작아질수록 좋은지, 커질수록 좋은지를 설정한다.
    #
    # loss, val_loss
    # → 작을수록 좋음
    # → mode='min'
    #
    # accuracy, val_accuracy, R2 등
    # → 클수록 좋음
    # → mode='max'
    #
    # mode='auto'로 하면 Keras가 이름을 보고 자동 판단한다.


    patience=10,
    # val_loss가 개선되지 않아도
    # 몇 epoch까지 더 기다릴지를 의미한다.
    #
    # 예:
    #
    # epoch 20 : val_loss = 3000  ← 현재 최고
    # epoch 21 : val_loss = 3010  ← 개선 없음 1회
    # epoch 22 : val_loss = 3020  ← 개선 없음 2회
    # ...
    # epoch 30 : val_loss = 3050  ← 개선 없음 10회
    #
    # 이런 식으로 연속해서 10 epoch 동안
    # 더 좋은 val_loss가 나오지 않으면 학습을 중단한다.
    #
    # 주의:
    # "10번째 전의 값을 가져온다"는 뜻이 아니다.
    #
    # patience의 단위는 batch나 step이 아니라 "epoch"이다.


    restore_best_weights=True
    # EarlyStopping이 학습을 중단했을 때
    # 마지막 epoch의 가중치를 사용하는 것이 아니라,
    #
    # monitor 값이 가장 좋았던 epoch의
    # W, b를 다시 복구한다.
    #
    # 예:
    #
    # epoch 20 : val_loss = 3000
    # epoch 21 : val_loss = 2900
    # epoch 22 : val_loss = 2800  ← 가장 좋음
    # epoch 23 : val_loss = 2850
    # ...
    # epoch 32 : val_loss = 3100 → EarlyStopping
    #
    # restore_best_weights=True라면
    # epoch 32의 W,b를 사용하는 것이 아니라
    # epoch 22의 W,b로 되돌아간다.
)


# ============================================================
# 훈련 시간 측정 시작
# ============================================================

start_time = time.time()
# 현재 시간을 초(second) 단위로 반환한다.
#
# 학습이 끝난 뒤 다시 time.time()을 호출해서
# 두 시간의 차이를 구하면 전체 훈련 시간을 측정할 수 있다.

# ============================================================
# 모델 훈련
# ============================================================

hist = model.fit(

    x_train,
    # 모델의 입력 데이터
    #
    # 예:
    # x_train.shape = (1000, 10)
    #
    # 데이터 1000개,
    # 각 데이터마다 feature가 10개 있다는 의미

    y_train,
    # x_train에 대응되는 정답 데이터
    #
    # 회귀 문제라면 실제 y값,
    # 분류 문제라면 class label 등이 들어간다.

    epochs=100,
    # 전체 훈련 데이터를 최대 몇 번 반복해서 학습할지 설정
    #
    # epochs=100
    # → 전체 x_train을 최대 100바퀴 학습한다.
    #
    # 하지만 EarlyStopping이 작동하면
    # 100 epoch 전에 학습이 종료될 수도 있다.

    batch_size=32,
    # 가중치 W와 편향 b를
    # 한 번 업데이트할 때 사용할 데이터 개수
    #
    # batch_size=32이면
    #
    # 32개 데이터
    # ↓
    # 예측
    # ↓
    # loss 계산
    # ↓
    # gradient 계산
    # ↓
    # optimizer가 W,b 업데이트
    #
    # 를 반복한다.
    #
    # 예를 들어 실제 train 데이터가 1000개라면
    #
    # 1000 / 32 ≈ 31.25
    #
    # 즉 한 epoch당 약 32번의
    # 가중치 업데이트가 일어난다.
    #
    # 일반적으로:
    #
    # batch_size 작음
    # → 업데이트 횟수 많음
    # → gradient 변동성이 큼
    #
    # batch_size 큼
    # → 여러 데이터의 gradient를 평균
    # → 비교적 안정적인 업데이트

    verbose=0,
    # 학습 진행 상황을 얼마나 출력할지 설정
    #
    # verbose=0
    # → 출력하지 않음
    #
    # verbose=1
    # → progress bar 형태로 출력
    #
    # verbose=2
    # → epoch마다 한 줄씩 출력

    validation_split=0.2,
    # x_train, y_train 중
    # 20%를 validation 데이터로 사용한다.
    #
    # 즉 x_train 전체를 실제 학습에 다 사용하는 것이 아니다.
    #
    # 예:
    #
    # x_train = 1000개
    #
    # 실제 학습 데이터
    # 1000 × 0.8 = 800개
    #
    # validation 데이터
    # 1000 × 0.2 = 200개
    #
    # validation 데이터는
    # W,b를 역전파로 직접 학습하는 데 사용하지 않는다.
    #
    # 대신 각 epoch가 끝난 뒤
    # 현재 모델이 새로운 데이터에 얼마나 잘 맞는지
    # val_loss 등을 계산하는 데 사용한다.
    #
    # 그리고 현재 EarlyStopping에서는
    # 이 val_loss를 보고 학습 중단 시점을 결정한다.

    callbacks=[es]
    # 학습 과정 중 실행할 callback 함수들을 지정한다.
    #
    # 여기서는 위에서 만든
    # EarlyStopping 객체 es를 전달한다.
    #
    # 따라서 model.fit이 진행되는 동안
    # 매 epoch가 끝날 때마다 es가 val_loss를 확인한다.
)


# ============================================================
# 훈련 시간 측정 종료
# ============================================================

end_time = time.time()

print("훈련 시간 :", end_time - start_time, "초")

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
results = model.predict(x , batch_size = 32)

# round(x, 2) : 소수점 두 번째 자리까지 반올림하여 출력합니다.
print("걸린 시간: ", round(end_time - start_time, 2), "초")

print("================= history =================")
print(hist)
print("================= hist.history =================")
print(hist.history)
print("================= loss =================")
print(hist.history['loss'])
print("================= val_loss =================")
print(hist.history['val_loss'])
print("==================================")

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) # 그래프 시각화를 위한 캔버스 크기 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트(맑은 고딕) 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스(-) 기호 깨짐 방지

# x축(시간 순서)을 생략하고 y값(loss)만 넣으면 자동으로 순서대로 그려집니다.
plt.plot(hist.history['loss'][2:], c='red', label='loss') # 훈련 손실(loss) 선 그래프
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss') # 검증 손실(val_loss) 선 그래프

plt.legend(loc='upper right') # 그래프 우측 상단에 범례(라벨) 표시
plt.title('캘리포니아 주택 가격 데이터셋 - Loss 그래프 (EarlyStopping 적용)')
plt.xlabel('Epoch') # x축 이름
plt.ylabel('Loss') # y축 이름 (코드 내 중복 오류 수정: xlabel -> ylabel)
plt.grid() # 그래프 배경에 격자 표시
plt.show() # 그래프 출력