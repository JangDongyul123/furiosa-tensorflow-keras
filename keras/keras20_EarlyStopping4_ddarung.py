# [실습] 과적합(Overfitting) 이해하기 - 따릉이 대여량 예측
# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddarung/" # 재사용성을 위해 경로 분리 저장
 
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
# index_col=0 : 첫 번째 컬럼(id)은 데이터가 아닌 인덱스로 취급하여 분석에서 제외함

print(train_csv) # [1459 rows x 10 columns] 
print(train_csv.shape)
print(train_csv.columns)
print(train_csv.info())

# [결측치 처리] 1. 결측치가 포함된 행 삭제 (dropna)
train_csv = train_csv.dropna()
print(train_csv) # 삭제 후: [1328 rows x 10 columns]

# train_csv에서 독립변수(x)와 종속변수(y) 분리
x = train_csv.drop(['count'], axis=1) # 예측 목표인 'count' 컬럼을 제외하여 입력 데이터(x) 생성
print("x: ", x) # [1328 rows x 9 columns] 

y = train_csv['count'] # 예측 목표 데이터(y) 추출
print(y)
print(y.shape) # (1328,)

# id 컬럼을 제외하면 x의 피처(컬럼)는 9개가 됩니다. id는 단순 식별자이므로 y 예측에 필요 없습니다.


test_csv = pd.read_csv(path + "test.csv", index_col=0) # 최종 예측에 사용할 평가용 데이터 (model.predict)
print(test_csv) #[715 * 9] 
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())


submission = pd.read_csv(path+"submission.csv", index_col=0) # 예측 결과를 저장하여 제출할 파일 양식
print(submission) # NaN은 결측치를 의미, [715 rows x 1 columns] 
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
          batch_size=32,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])        # EarlyStopping 콜백 적용

#4. 평가, 예측

print("========================================")
y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  # 사용자가 직접 정의한 RMSE(Root Mean Squared Error) 함수
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


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
plt.title('데이콘 따릉이 예측 데이터셋 - Loss 그래프')
plt.xlabel('Epoch') # x축
plt.ylabel('Loss') # y축
plt.grid() # 격자 추가
plt.show() # 출력