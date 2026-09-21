import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# 1. NPY 데이터 불러오기 (초고속 로딩)
start_time_load = time.time()

np_path = './_data/brain_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy')

print(f"npy 로드 시간 : {round(time.time() - start_time_load, 2)} 초")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

early_stopping = EarlyStopping(
    monitor='val_acc', 
    mode='max', 
    patience=130, 
    restore_best_weights=True,
    verbose=1
)

start_time_train = time.time()
model.fit(
    x_train, y_train, 
    epochs=200, 
    batch_size=13,
    validation_split=0.3,
    callbacks=[early_stopping],
    verbose=1
)
end_time_train = time.time()

# 4. 평가 및 결과 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss[0]}")
print(f"acc : {loss[1]}")

y_pred = np.round(model.predict(x_test))
print(f"accuracy_score : {accuracy_score(y_test, y_pred)}")
print(f"훈련 소요 시간 : {round(end_time_train - start_time_train, 2)} 초")