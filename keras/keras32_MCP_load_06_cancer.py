from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer # 유방밤 관련 데이터

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)

#x = datasets.data
x = datasets['data'] # 이렇게도 가능하다.
y = datasets.target

# Number of Instances: 569, number of Attributes: 30
#  569행 30열
print(x.shape) # (569,30)
print(y.shape) # (569, )
print(type(x)) # numpy


print(y) # 0과 1 데이터

# 0과 1의 개수가 몇개인지 찾아보기 -numpy
print(np.unique(y)) # [0 1]
#분류 데이터 10만개 데이터 중 0 또는 1이 아니라 다른 이상치가 있을 수 있으므로
print(np.unique(y, return_counts=True)) 
#array([0,1]), array([212, 357])

# 0과 1의 개수가 몇개인지 찾아보기 - pandas
print(pd.DataFrame(y).value_counts())
# 1 357
# 0 212
print(pd.Series(y).value_counts())
# 1 357
# 0 212

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7, 
    random_state=333,
    stratify=y, 
    # 분류에서는 startify = y 를 사용한다. 그 이유가 뭐지???
    # y를 기준으로 0 과 1의 분포를 동일하게 잘라줘라?
    )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

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



print(np.unique(y_train, return_counts=True))
# 만약 (array([0, 1]), array([1480, 249], dtype=int64)) 라ㄱ호 갖ㅇ하자
# 훈련시키는 데이터는 불균형하면 문제가 생긴다.
# 학습시키는 데이터니까 더 많은 숫자인 0에 치우친다.
# 훈련 데이터는 비율이 유지되어야 한다.
# 그래서 대회에서 데이터를 불균형하게 주면 부족한 라벨 데이터를 증폭시켜서 비율을 맞춘다.

print(np.unique(y_test, return_counts=True))
# 만약 (array([0, 1]), array([6, 107], dtype=int64)) 라고 가정하자
# 시험에 쓰이는 데이터는 비율이 무너져도 상관없다.
# 학습 데이터가 아니라서. 오로지 구분에 쓰이는 데이터이기에.


# startify = y를 사용한 다음
print(np.unique(y_train, return_counts=True))
# 만약 (array([0, 1]), array([148, 250], dtype=int64))

print(np.unique(y_test, return_counts=True))
# 만약 (array([0, 1]), array([64, 107], dtype=int64))

print(x_train.shape) #(398, 30)
print(x_test.shape)  #(171, 30)
print(y_train.shape) #(398, )
print(y_test.shape)  #(171, )



#2. 모델 구성
model = load_model('C:\study\_save\keras31\k31_0914_1440-_0027-0.1438.keras')

print("================================")
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)


r2 = r2_score(y_test, results)

print("loss : ", loss) # loss와 accuracy가 나온다.
print("loss : ", loss[0]) # loss
print("loss : ", round(loss[1], 4)) # accuracy

print("r2_score : ", r2)
print("results : ", results)

def rmse(y_test, results):
    return np.sqrt(mean_squared_error(y_test, results))

rmse = rmse(y_test, results)
print("rmse : ", rmse)

y_pred = model.predict(x_test)
print(y_pred)
print("================================")
'''
y_pred 값은 sigmoid 값으면 0~1 사이 값.
최종 wx+b 가 음수면 sigmoid(wx + b) 가 0.5 미만
wx+b가 양수면 sigmoid(wx + b) 가 0.5 이상
[8.6131716e-01]
[9.8821718e-01]
[3.3848113e-10]

여기서 0.5 기준으로 반올림 해주면, 0과 1로 값이 나온다.

'''

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, np.round(y_pred))
#y_test는 0과 1사이로 되어있으나, y_pred는 sigmoid를 통과해서 0~1 사이로 나와서 에러가 뜬 것
#그래서 np.arround를 써야한다.

print(y_pred[:10])
y_pred = np.round(y_pred)
print(y_pred[:10])

print("acc_score: ", accuracy_score)