import numpy as np
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# 1. NPY 파일 로드 (속도 측정)
start_time_load = time.time()

np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

print(f"npy 로드 시간 : {round(time.time() - start_time_load, 2)} 초")

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 메모리 확보(선택사항)
# del x_train, y_train... 등은 필요 시 사용

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(Conv2D(32, (5,5), activation='relu'))
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

# ModelCheckpoint 파일 경로 지정
date = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = f'./_save/keras45/k_{date}_' + '{epoch:04d}_{val_acc:.4f}.keras'

mcp = ModelCheckpoint(
    monitor='val_acc', 
    mode='max', 
    save_best_only=True, 
    filepath=filepath,
    verbose=1
)

start_time_train = time.time()
model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, mcp],
    verbose=1
)
end_time_train = time.time()

# 4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss[0]}")
print(f"acc : {loss[1]}")

y_pred = np.round(model.predict(x_test))
print(f"accuracy_score : {accuracy_score(y_test, y_pred)}")
print(f"훈련 소요 시간 : {round(end_time_train - start_time_train, 2)} 초")