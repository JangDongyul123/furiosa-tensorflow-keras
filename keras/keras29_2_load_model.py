#29 save 실습

# [실습] Min-Max Scaler 이해하기 - 캘리포니아 주택 가격 데이터셋
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



print(np.min(x_train), np.max(x_train))

# exit()

#2. 모델구성
# model = Sequential()
# model.add(Dense(16, input_dim=8 , activation= 'relu'))
# model.add(Dense(8, activation= 'relu'))
# model.add(Dense(4, activation= 'relu'))
# model.add(Dense(2, activation= 'relu'))
# model.add(Dense(1))

# model.summary()

path = './_save/keras29/'

# model.save(path + 'keras29_1_save_model.keras')

model = load_model(path + 'keras29_1_save_model.keras')
# 저장된 모델을 가져온다.

model.summary()


# exit()

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
# 참고: round(x, 2)는 소수점 둘째 자리까지 반올림합니다.

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