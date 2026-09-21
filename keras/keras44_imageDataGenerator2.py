import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# 1. 데이터 준비
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=160,           # 전체 데이터 한 번에 가져오기
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=120,           # 전체 데이터 한 번에 가져오기
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
)

# 데이터 분리 (x, y)
x_train, y_train = train_data[0][0], train_data[0][1]
x_test, y_test = test_data[0][0], test_data[0][1]

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

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
model.add(Dense(1, activation='sigmoid')) # 이진 분류

model.summary()

# 3. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

early_stopping = EarlyStopping(
    monitor='val_acc', 
    mode='auto', 
    patience=130, 
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=200, 
    batch_size=13,
    validation_split=0.3,
    callbacks=[early_stopping],
    verbose=1
)
end_time = time.time()

# 4. 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss[0]}")
print(f"acc : {loss[1]}")

y_pred = np.round(model.predict(x_test))
print(f"accuracy_score : {accuracy_score(y_test, y_pred)}")
print(f"소요 시간 : {round(end_time - start_time, 2)} 초")