import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print("train shape:", x_train.shape, y_train.shape)
print("val shape:", x_val.shape, y_val.shape)
print("test shape:", x_test.shape, y_test.shape)

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1,
    rotation_range=15,
    # zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest',
)

augment_size = 40000

# 6만개 중에 4만개 랜덤 뽑기
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 차원 증가 (흑백이므로 1)
x_augmented = x_augmented.reshape(-1, x_augmented.shape[1], x_augmented.shape[2], 1)

# 증강 데이터 생성 (datagen 거치면서 0~1 스케일링됨)
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
    seed=42,
)
x_augmented = next(x_augmented)[0]

# 원본 데이터도 차원 증가 및 0~1 스케일링 적용
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype(np.float32) / 255.0
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype(np.float32) / 255.0

# 원본과 증강 데이터 결합 (val 데이터는 순수하게 유지)
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# 데이터 섞기 (원본과 증강 데이터가 골고루 섞이도록 명시적으로 셔플)
shuffle_idx = np.random.permutation(len(x_train))
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print("증강 후 x_train shape:", x_train.shape)
print("증강 후 y_train shape:", y_train.shape)

# 원핫 인코딩
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 실험 목적의 Conv1D 입력이다. 이미지의 세로축만 timestep으로 취급하므로
# 일반적인 이미지 분류에서는 공간 정보를 함께 쓰는 Conv2D가 더 자연스럽다.
# Conv1D 입력을 위해 3차원으로 변환
x_train = x_train.reshape(-1, 28, 28)
x_val = x_val.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)


from tensorflow.keras.optimizers import Adam

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(28, 28), padding='same'))
model.add(MaxPooling1D())

model.add(Conv1D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
learning_rate = 0.001 # 디폴트
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True
)

start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=128, 
    validation_data=(x_val, y_val), 
    callbacks=[es]
)
end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test_arg = np.argmax(y_test, axis=1)

print('accuracy_score :', accuracy_score(y_test_arg, y_pred))
print('훈련 소요 시간 :', round(end_time - start_time, 2), '초')
