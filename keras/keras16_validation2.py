import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# [실습] 전체 데이터를 훈련(8), 검증(4), 테스트(4) 비율로 분할하기

# 1단계: 먼저 Train(8개)과 나머지(Test+Validation, 8개)를 5:5로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

print(x_train.shape)
print(x_test.shape)

# 2단계: 나머지 8개를 다시 Validation(4개)과 Test(4개)로 5:5 분할
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 300, batch_size= 12, verbose = 1, validation_data = (x_validation,y_validation))

#4. 평가, 예측
loss = np.sqrt(model.evaluate(x_test, y_test))
results = model.predict(x_test)