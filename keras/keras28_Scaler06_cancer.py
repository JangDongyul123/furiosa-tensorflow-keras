from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
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
model = Sequential()
model.add(Dense(7000, input_dim = x.shape[1], activation = 'relu'))
model.add(Dense(9000, activation = 'relu'))
model.add(Dense(7000, activation = 'relu'))
model.add(Dense(3000, activation = 'relu'))
model.add(Dense(2000, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(1, activation='sigmoid')) 
# 이진 분류에서는 마지막 레이어에 sigmoid를 사용한다. 확률을 반환한다.
# 외워라
# sigmoid는 입력값에 상관없이 항상 0과 1 사이의 값을 반환합니다.
# activation은 활성화 함수 또는 함정함수라고 부른다.

#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy']) 
# 외워라 이진분류에서는 loss를 'binary_crossentropy' 를 사용한다.
# 이진분류는 0이냐 1이냐를 찾는다.
# loss는 역전파와 w 갱신을 위해 사용한다.
# metrix는 accuracy라는 보조지표를 이용한다.
# accuracy가 0.92가 나오면 적중률 92%라는 뜻 
# val_acc 값이  

es = EarlyStopping(
    monitor='val_loss',  # 훈련 손실(loss)을 관찰합니다.
    patience=100,     # 20 epoch 동안 성능 향상이 없으면 멈춥니다.
    mode='min',      # loss는 낮을수록 좋으므로 최소(min)값을 추적합니다.
    restore_best_weights=True #  
)

start_time = time.time()

model.fit(x_train,
        y_train,
        epochs = 30000, 
          batch_size = x.shape[0], 
          verbose = 0, 
          validation_split=0.3,
          callbacks = [es]
          )
end_time = time.time()

#4. 평가 예측

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
y.pred = np.round(y_pred)
print(y_pred[:10])

print("acc_score: ", accuracy_score)