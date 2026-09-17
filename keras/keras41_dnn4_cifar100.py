# 0.48 스코어
# 걸린 시간 292.76초

import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

# =================================================================================
# 1. 데이터 준비
# =================================================================================
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 스케일링
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_test = (x_test.astype(np.float32) - 127.5) / 127.5

# OneHotEncoding
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# =================================================================================
# 2. 모델 구성
# =================================================================================
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D)
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# =================================================================================
# 3. 컴파일, 훈련
# =================================================================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es], verbose=1)
end_time = time.time()

# =================================================================================
# 4. 평가, 예측
# =================================================================================
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"최종 Accuracy : {acc:.4f}")
print(f"걸린 시간 : {round(end_time - start_time, 2)}초")