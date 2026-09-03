# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddarung/" #재사용성을 위해 경로 분리
 
train_csv = pd.read_csv(path + "train.csv", index_col= 0) 
#index_col 이 컬럼은 인덱스니 제외하라고 알려주는 파라미터
print(train_csv) # [1459 * 10] 
print(train_csv.shape)
print(train_csv.columns)
print(train_csv.info())

# 결측치 처리 1. 삭제
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

#train_csv에서 x와 y를 분리
x = train_csv.drop(['count'], axis =1) # count 열(컬럼)을 지운다.
print("x: ", x) # [1328 * 9] 

y = train_csv['count'] 
print(y)
print(y.shape) # (1328,)

#id 컬럼을 제외하면 컬럼이 10개가 된다. id 컬럼은 y에게 영향을 주지 않는다.


test_csv = pd.read_csv(path + "test.csv", index_col= 0) #model.predict()에 넣을 값
print(test_csv) #[715 * 9] 
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())


submission = pd.read_csv(path+"submission.csv" , index_col= 0) #여기에 model.predict(submission) 값을 넣는다.
print(submission) # NaN은 결측치를 의미, [715 * 9] 
print(submission.shape)
print(submission.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

#4. 평가, 예측

print("========================================")
y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)