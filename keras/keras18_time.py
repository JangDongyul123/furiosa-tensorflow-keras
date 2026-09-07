# [실습] 모델 학습에 걸린 시간 측정하기

# 이 링크에서 .csv파일을 다운로드 받으셔야 합니다.
# https://www.kaggle.com/competitions/bike-sharing-demand/data


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)  #  [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv)  #  [6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

print(submission)  #  [6493 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)

print(train_csv.info())
print(test_csv.info())
print(submission.info())

print(train_csv.describe())
############################### 결측치 확인 ###############################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())


############################### x,y 분리 ###############################
x = train_csv.drop(['casual','registered','count'], axis = 1)
print(x)
y = train_csv['count']
print(y, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim =x.shape[1] , activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# 훈련 시간 측정 시작
start_time = time.time() # 현재 시스템 시간을 초 단위로 반환하여 기록

model.fit(x_train, 
          y_train, 
          epochs=100, 
          batch_size=16,
          verbose=1,
          validation_split=0.33
          )

# 훈련 시간 측정 종료
end_time = time.time() # 학습 완료 후의 시스템 시간을 기록
# 시작 시간과 종료 시간의 차이를 계산하여 전체 훈련에 소요된 시간을 구할 수 있습니다.

#4. 평가, 에측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

rmsle = np.sqrt(mean_squared_log_error(y_test,y_pred))
print("rmsle : ", rmsle)

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)

# round(x, 2) : 소수점 두 번째 자리까지 반올림하여 출력합니다.
print("걸린 시간: ", round(end_time - start_time, 2), "초")

"""
#5. 제출
y_submit = model.predict(test_csv)
print(y_submit.shape)

submission['count'] = y_submit
submission.to_csv(path + "/submit/submission.csv")
"""

