import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 준비 (Fashion MNIST)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2차원 변환
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)

# 타겟 데이터 원핫 인코딩
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# 2. 모델 구성 (함수형 API)
inputs = Input(shape=(784,))
d1 = Dense(512, activation='relu')(inputs)
d2 = Dense(256, activation='relu')(d1)
d3 = Dense(128, activation='relu')(d2)
d4 = Dense(128, activation='relu')(d3)
drop1 = Dropout(0.4)(d4)

d5 = Dense(64, activation='relu')(drop1)
d6 = Dense(64, activation='relu')(d5)
d7 = Dense(32, activation='relu')(d6)
d8 = Dense(32, activation='relu')(d7)
drop2 = Dropout(0.3)(d8)

d9 = Dense(16, activation='relu')(drop2)
d10 = Dense(16, activation='relu')(d9)
drop3 = Dropout(0.2)(d10)

outputs = Dense(10, activation='softmax')(drop3)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

# 3. 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

early_stopping = EarlyStopping(
    monitor='val_acc', 
    mode='max', 
    patience=25, 
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
end_time = time.time()

# 4. 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss[0]}")
print(f"acc : {loss[1]}")

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"accuracy_score : {accuracy_score(y_true, y_pred)}")
print(f"총 소요 시간 : {round(end_time - start_time, 2)} 초")