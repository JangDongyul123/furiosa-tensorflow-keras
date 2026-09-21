import numpy as np
import time
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 (개/고양이)
start_time = time.time()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = './_data/image/cat-dog/training_set/'
path_test = './_data/image/cat-dog/test_set/'

train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=8005,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=2023,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

x_train, y_train = train_data[0][0], train_data[0][1]
x_test, y_test = test_data[0][0], test_data[0][1]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# 2. 모델링
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

# 3. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc', 
    mode='max', 
    patience=130, 
    restore_best_weights=True,
    verbose=1
)

date = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = f'./_save/keras44/k_{date}_' + '{epoch:04d}_{val_acc:.4f}.keras'

mcp = ModelCheckpoint(
    monitor='val_acc', 
    mode='max', 
    save_best_only=True, 
    filepath=filepath,
    verbose=1
)

train_start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=32,
    validation_split=0.1,
    callbacks=[es, mcp],
    verbose=1
)
train_end_time = time.time()

# 4. 평가
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss}")
print(f"acc : {acc}")

y_pred = np.round(model.predict(x_test))
print(f"accuracy_score : {accuracy_score(y_test, y_pred)}")
print(f"전체 소요 시간 : {round(train_end_time - start_time, 2)} 초")