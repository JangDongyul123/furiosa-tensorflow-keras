#save 실습
#30-1 카피

# [실습] Min-Max Scaler 이해하기 - 캘리포니아 주택 가격 데이터셋

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

path = './_save/keras30/'


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

#2. 모델구성

model = load_model(path + 'keras30_mcp1.keras')

#3. 컴파일, 훈련

#4. 평가, 예측

print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
y_pred = model.predict(x_test, batch_size = 32)

print("걸린 시간: ", round(end_time- start_time, 2), "초")

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


rmse = RMSE(y_test, y_pred)z
print('mse : ', mse)
print('rmse : ', rmse   )

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)