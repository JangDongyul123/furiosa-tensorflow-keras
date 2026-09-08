from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터

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
    # 분류 문제이므로 타겟(y)의 클래스 비율을 훈련/테스트 셋에 동일하게 분배하기 위해 stratify=y를 사용합니다.
    )

print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# stratify=y 적용 후 실제 분할된 비율 확인
# 훈련 및 테스트 데이터의 라벨 비율이 원본과 유사하게 분할됨을 확인 (학습 치우침 방지)
print("Train target counts:", np.unique(y_train, return_counts=True))
print("Test target counts:", np.unique(y_test, return_counts=True))

print(x_train.shape) #(398, 30)
print(x_test.shape)  #(171, 30)
print(y_train.shape) #(398, )
print(y_test.shape)  #(171, )


#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = x.shape[1], activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation='sigmoid')) 
# 이진 분류의 마지막 레이어 활성화 함수는 'sigmoid'를 사용하여 0~1 사이의 확률값을 반환합니다.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) 
# 이진 분류의 loss 함수는 'binary_crossentropy'를 사용하며,
# 평가 보조 지표로 'accuracy'(정확도)를 추가합니다.

es = EarlyStopping(
    monitor='val_loss',          # 관찰할 지표 (검증 손실)
    patience=1000,               # 1000 epoch 동안 개선이 없으면 조기 종료
    mode='min',                  # 관찰 지표가 최소가 될 때 최적
    restore_best_weights=True    # 조기 종료 시 가장 성능이 좋았던 가중치로 복구
)

start_time = time.time()

model.fit(x_train,
        y_train,
        epochs = 3000, 
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

print("================================")
# y_pred는 sigmoid를 통과하여 0~1 사이의 확률값으로 나옵니다.
# 0.5를 기준으로 반올림(np.round)하여 0과 1의 이진 정답으로 변환합니다.

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, np.round(y_pred))

print("sigmoid 예측값 예시:\n", y_pred[:10])
y_pred = np.round(y_pred)
print("반올림 후 예측값 예시:\n", y_pred[:10])

print("최종 acc_score: ", acc_score)