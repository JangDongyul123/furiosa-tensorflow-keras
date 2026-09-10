# [실습] Min-Max Scaler 이해하기 - 캐글 자전거 수요 예측 데이터셋


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)  #  [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 최종 예측용 평가 데이터

print(test_csv)  #  [6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0) # 예측 결과를 기입할 제출 양식

print(submission)  #  [6493 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)

print(train_csv.info())
print(test_csv.info())
print(submission.info())

print(train_csv.describe())
# =================================================================================
# 결측치 확인
# =================================================================================
print(train_csv.isna().sum())
print(train_csv.isnull().sum())


# =================================================================================
# x, y 분리 (종속 변수 'count' 추출, 관련 없는 컬럼 'casual', 'registered' 제거)
# =================================================================================
x = train_csv.drop(['casual','registered','count'], axis = 1)
print(x)
y = train_csv['count']
print(y, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
# x_train 데이터를 보유한 데이터, 
# x_test를 미래 데이터라고 가정한다.
# 나중에 대회에서도 private 데이터가 아닌 우리가 보유한 데이터를 기준으로 스케일링할 것이 아닌가?
# 그러므로 x_train 데이터를 기준으로 스케일링한다.


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x.shape, y.shape)

print(x, y)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(6400, input_dim =x.shape[1] , activation = 'relu'))
model.add(Dense(6400, activation = 'relu'))
model.add(Dense(3200, activation = 'relu'))
model.add(Dense(1600, activation = 'relu'))
model.add(Dense(1)) 

# ============================================================
# EarlyStopping 설정
# ============================================================

es = EarlyStopping(
    monitor='val_loss',          # 관찰할 지표 (검증 손실)
    mode='min',                  # 관찰 지표가 최소가 될 때 최적
    patience=10,                 # 10 epoch 동안 개선이 없으면 조기 종료
    restore_best_weights=True    # 조기 종료 시 가장 성능이 좋았던 가중치로 복구
)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, 
          y_train, 
          epochs=1000, 
          batch_size=16000,
          verbose=1,
          validation_split=0.33,
          callbacks=[es]         # EarlyStopping 콜백 적용
          )

#4. 평가, 에측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

rmsle = np.sqrt(mean_squared_log_error(y_test,y_pred))
print("rmsle : ", rmsle)

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)

#5. 제출
y_submit = model.predict(test_csv)
print(y_submit.shape)

submission['count'] = y_submit
submission.to_csv(path + "/submit/submission.csv")

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

plt.figure(figsize=(9,6)) # 캔버스(판) 크기 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 폰트 설정
plt.rcParams['axes.unicode_minus'] = False 

plt.plot(hist.history['loss'][2:], c='red', label='loss') # 훈련 손실
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss') # 검증 손실

plt.legend(loc='upper right') # 우측 상단 범례 표시
plt.title('캐글 자전거 수요 예측 데이터셋 - Loss 그래프')
plt.xlabel('Epoch') # x축
plt.ylabel('Loss') # y축
plt.grid() # 격자 추가
plt.show() # 출력