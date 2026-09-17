# CPU 34초, GPU 76초

# [실습] Min-Max Scaler 이해하기 - 캘리포니아 주택 가격 데이터셋
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7, 
    test_size=0.3,
    random_state=333
    )


from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8 , activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

# exit()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
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

path = './_save/keras30/'
file_name = '_{epoch:04d}-{val_loss:.4f}.keras' # 04d는 4자리 정수, .4f는 소수점 4째자리까지
filepath = "".join([path, "k30_" ,date,"-", file_name])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)
# 최적의 모델을 저장

# 훈련 시간 측정 시작
start_time = time.time() # 현재 시스템 시간을 기록 (시작 시간)

hist = model.fit(x_train, y_train, 
          epochs=100, batch_size=32, verbose=1, validation_split=0.2,
          callbacks=[]
          )

end_time = time.time() # 학습 완료 후의 시스템 시간을 기록 (종료 시간)
print(end_time - start_time)
# model.load_weights(path+'keras29_5_save_weights2.weights.h5')

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
y_pred = model.predict(x_test, batch_size = 32)

print("걸린 시간: ", round(end_time- start_time, 2), "초")

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


rmse = RMSE(y_test, y_pred)
print('mse : ', mse)
print('rmse : ', rmse)

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)
