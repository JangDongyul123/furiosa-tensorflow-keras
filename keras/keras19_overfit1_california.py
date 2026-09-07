# [실습] 과적합(Overfitting) 이해하기 - 캘리포니아 주택 가격 데이터셋

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

# 훈련 시간 측정 시작
start_time = time.time() # 현재 시스템 시간을 기록 (시작 시간)

hist = model.fit(x_train, 
          y_train, 
          epochs=100, 
          batch_size=32,
          verbose=0,
          validation_split=0.2
          )

end_time = time.time() # 학습 완료 후의 시스템 시간을 기록 (종료 시간)

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
results = model.predict(x , batch_size = 32)

print("걸린 시간: ", round(end_time- start_time, 2), "초")
# round(x, 2) 함수는 소수점 둘째 자리까지 반올림하여 출력합니다.

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

plt.figure(figsize=(9,6)) # 그래프 시각화를 위한 캔버스(판) 크기 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트(맑은 고딕) 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스(-) 기호 깨짐 방지

# x축(시간 순서)을 생략하고 y값(loss)만 넣으면 자동으로 순서대로 그려집니다.
plt.plot(hist.history['loss'][2:], c='red', label='loss') # 훈련 손실(loss) 선 그래프
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss') # 검증 손실(val_loss) 선 그래프

plt.legend(loc='upper right') # 그래프 우측 상단에 범례(라벨) 표시
plt.title('캘리포니아 주택 가격 데이터셋 - Loss 그래프')
plt.xlabel('Epoch') # x축 이름
plt.ylabel('Loss') # y축 이름 (코드 내 중복 오류 수정: xlabel -> ylabel)
plt.grid() # 그래프 배경에 격자 표시
plt.show() # 그래프 출력