# [실습] Min-Max Scaler 이해하기

# [실습] 과적합(Overfitting) 이해하기 - 캘리포니아 주택 가격 데이터셋
from sklearn.metrics import mean_squared_error, r2_score
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


'''
MinMaxScaler

원값 - Min
---------
Max - Min

모든 값은 0 ~ 1 사이로 수렴하는 식으로 scaling 해준다.

예) X의 최대값이 10000이면 1 , 최소값이 -1이면 0이 됨
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0

# exit()

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
y_pred = model.predict(x_test, batch_size = 32)

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

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


rmse = RMSE(y_test, y_pred)
print('mse : ', mse)
print('rmse : ', rmse   )

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)