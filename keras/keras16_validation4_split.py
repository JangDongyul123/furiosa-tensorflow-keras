import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

print(x_train.shape)
print(x_test.shape)

print(x_train.shape)
print(x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, 
            epochs = 300, 
            batch_size= 12, 
            verbose = 1, 
            # validation_data=(x_validation, y_validation) # 직접 검증 데이터를 넣는 방식
            validation_split=0.33, # 훈련 데이터 중 33%를 자동으로 검증 데이터(Validation)로 할당
            )   

#4. 평가, 예측
loss = np.sqrt(model.evaluate(x_test, y_test))
results = model.predict(x_test)