# [실습] Min-Max Scaler 이해하기 - 캐글 자전거 수요 예측 데이터셋


from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
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

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
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


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = load_model('C:\study\_save\keras31\k31_0914_1434-_0003-32480.0508.keras')

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
