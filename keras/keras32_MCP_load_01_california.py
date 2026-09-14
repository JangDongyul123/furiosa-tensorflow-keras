# [실습] Min-Max Scaler 이해하기 - 캘리포니아 주택 가격 데이터셋
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
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
    random_state=333)


# =================================================================================
# [ MinMaxScaler ]
# 공식: (x - Min) / (Max - Min)
# 특징: 모든 값을 0 ~ 1 사이의 값으로 변환하여 스케일링합니다.
# 예시: 데이터의 최대값이 10000이면 1로, 최소값이 -1이면 0으로 변환됩니다.
# =================================================================================

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# =================================================================================
# [ 스케일러 학습 (Fit) 주의사항 ]
# x_train 데이터만 이용해서 스케일링 기준(Min/Max, Mean/Std 등)을 학습합니다.
# x_val, x_test, 그리고 실전(Kaggle 등)의 미래 데이터는
# 오직 x_train에서 학습한 동일한 기준으로 transform만 수행해야 합니다.
# (Validation/Test 데이터의 정보가 스케일러에 미리 반영되는 것을 방지하기 위함)
# =================================================================================


# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))
# 0.0 1.0


from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# scaler = StandardScaler()

# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()



x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = load_model('C:\study\_save\keras31\k30_0914_1404-_0010-0.3844.keras')

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 32)
print("========================================")
y_pred = model.predict(x_test, batch_size = 32)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


rmse = RMSE(y_test, y_pred)
print('mse : ', mse)
print('rmse : ', rmse   )

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)