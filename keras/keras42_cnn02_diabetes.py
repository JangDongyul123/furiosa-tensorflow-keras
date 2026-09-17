# CPU 6.16초, GPU 2.8초

# [실습] Min-Max Scaler 이해하기 - 당뇨병 데이터셋


import time
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=333);

# =================================================================================
# [ MinMaxScaler ]
# 공식: (x - Min) / (Max - Min)
# 특징: 모든 값을 0 ~ 1 사이의 값으로 변환하여 스케일링합니다.
# 예시: 데이터의 최대값이 10000이면 1로, 최소값이 -1이면 0으로 변환됩니다.
# =================================================================================

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
# scaler = StandardScaler()

# scaler = MaxAbsScaler()

scaler = MinMaxScaler()
scaler.fit(x_train)
# =================================================================================
# [ 스케일러 학습 (Fit) 주의사항 ]
# x_train 데이터만 이용해서 스케일링 기준(Min/Max, Mean/Std 등)을 학습합니다.
# x_val, x_test, 그리고 실전(Kaggle 등)의 미래 데이터는
# 오직 x_train에서 학습한 동일한 기준으로 transform만 수행해야 합니다.
# (Validation/Test 데이터의 정보가 스케일러에 미리 반영되는 것을 방지하기 위함)
# =================================================================================


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x.shape, y.shape)

print(x, y)

#2. 모델 구성

# model = Sequential()
# model.add(Dense(1, input_dim = x.shape[1]))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(1))

input1 = Input(shape=(x.shape[1],))
dense1 = Dense(1)(input1)
dense2 = Dense(10)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(10)(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(10)(drop2)
drop3 = Dropout(0.2)(dense4)
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1)

#########################################





# ============================================================
# EarlyStopping 설정
# ============================================================

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = Adam(learning_rate=0.9))

es = EarlyStopping(
    monitor='val_loss',
      patience=10,
    mode='min',
    restore_best_weights= True,
    verbose=1
)

############# mcp 세이브 파일명 만들기 #############

import datetime
date = datetime.datetime.now()
print(date) # 2026-09-14 11:42:07 .201728
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #month day, hour, minutes
print(date)
print(type(date)) #<class 'str'>

path = './_save/keras31/'
file_name = '_{epoch:04d}-{val_loss:.4f}.keras' # 04d는 4자리 정수, .4f는 소수점 4째자리까지
filepath = "".join([path, "k31_" ,date,"-", file_name])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, 
          y_train, 
          epochs= 100, 
          batch_size=1000, 
          verbose=1, 
          validation_split=0.33,
          callbacks=[]         # EarlyStopping 콜백 적용
          )
end_time = time.time()
print(end_time - start_time)
#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
results = model.predict(x)

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
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# y값(loss)만 넣으면 자동으로 순서대로 그려집니다.
plt.plot(hist.history['loss'][2:], c='red', label='loss') # 훈련 손실(loss) 
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss') # 검증 손실(val_loss)

plt.legend(loc='upper right') # 그래프 우측 상단에 범례(라벨) 표시
plt.title('당뇨병 데이터셋 - Loss 그래프')
plt.xlabel('Epoch') # x축 이름
plt.ylabel('Loss') # y축 이름 (오류 수정: xlabel -> ylabel)
plt.grid() # 배경에 격자 표시
plt.show() # 그래프 출력