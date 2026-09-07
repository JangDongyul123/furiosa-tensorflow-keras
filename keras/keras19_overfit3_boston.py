# [실습] 과적합(Overfitting) 이해하기 - 보스턴 주택 가격 데이터셋

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

#1. 데이터 구성
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, test_size=0.3)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(13, input_dim = 13))
model.add(Dense(1))

#3. 컴파일, 훈련
hist = model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, 
          y_train, 
          verbose=1,
          epochs=500, 
          batch_size =100,
          validation_split =0.33)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x)
# print("results: ", results)

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
plt.title('보스턴 주택 가격 데이터셋 - Loss 그래프')
plt.xlabel('Epoch') # x축
plt.ylabel('Loss') # y축
plt.grid() # 격자 추가
plt.show() # 출력