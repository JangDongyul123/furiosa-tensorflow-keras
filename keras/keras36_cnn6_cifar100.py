import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# =================================================================================
# 0. Seed
# =================================================================================

np.random.seed(333)
tf.random.set_seed(333)


# =================================================================================
# 1. 데이터
# =================================================================================

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
# x_train : (50000, 32, 32, 3)
# y_train : (50000, 1)

print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)
# x_test  : (10000, 32, 32, 3)
# y_test  : (10000, 1)


# =================================================================================
# 2. X Scaling
#
# 0 ~ 255
#    ↓
# -1 ~ 1
# =================================================================================

x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_test = (x_test.astype(np.float32) - 127.5) / 127.5


print("\nScaling 확인")
print("x_train min :", np.min(x_train))
print("x_train max :", np.max(x_train))


# =================================================================================
# 3. Y 원본 저장
#
# 나중에 sklearn accuracy_score 계산할 때 사용
# =================================================================================

y_train_original = y_train.copy()
y_test_original = y_test.copy()


# =================================================================================
# 4. OneHotEncoding
#
# CIFAR100
#
# 클래스:
# 0 ~ 99
#
# (50000, 1)
#      ↓
# (50000, 100)
# =================================================================================

ohe = OneHotEncoder(
    sparse_output=False
)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


print("\nOneHotEncoding")
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# (50000, 100)
# (10000, 100)


# =================================================================================
# 5. 모델 구성
# =================================================================================

model = Sequential()


# =================================================================================
# Block 1
#
# 입력:
# 32 × 32 × 3
#
# Conv:
# 32 × 32 × 32
#
# Conv:
# 32 × 32 × 32
#
# MaxPooling:
# 16 × 16 × 32
# =================================================================================

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        input_shape=(32, 32, 3)
    )
)

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2),

        # 2칸씩 이동
        # 즉 가로 / 세로 크기를 절반으로 줄임
        strides=(2, 2)
    )
)


# =================================================================================
# Block 2
#
# 입력:
# 16 × 16 × 32
#
# Conv:
# 16 × 16 × 64
#
# Conv:
# 16 × 16 × 64
#
# MaxPooling:
# 8 × 8 × 64
# =================================================================================

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
)


# =================================================================================
# Block 3
#
# 입력:
# 8 × 8 × 64
#
# Conv:
# 8 × 8 × 128
#
# Conv:
# 8 × 8 × 128
#
# MaxPooling:
# 4 × 4 × 128
# =================================================================================

model.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

model.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
)


# =================================================================================
# 6. Flatten + Dense
#
# 4 × 4 × 128
#
#     ↓ Flatten
#
# 2048
#
# =================================================================================

model.add(
    Flatten()
)


# 2048
#   ↓
# 128

model.add(
    Dense(
        128,
        activation='relu'
    )
)


# 128
#  ↓
# 64

model.add(
    Dense(
        64,
        activation='relu'
    )
)


# 64
# ↓
# 100개 클래스 확률

model.add(
    Dense(
        100,
        activation='softmax'
    )
)


# =================================================================================
# 모델 확인
# =================================================================================

model.summary()


# =================================================================================
# 7. EarlyStopping
# =================================================================================

# -------------------------------------------------------------------------
# 1단계
#
# epochs=30이기 때문에
# patience=50은 의미가 없음
#
# 여기서는 patience=10 사용
# -------------------------------------------------------------------------

es1 = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# -------------------------------------------------------------------------
# 2단계
# -------------------------------------------------------------------------

es2 = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# -------------------------------------------------------------------------
# 3단계
# -------------------------------------------------------------------------

es3 = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# =================================================================================
# 8. 1단계
#
# Loss:
# MSE
#
# Batch:
# 32
#
# 작은 Batch로 먼저 학습
# =================================================================================

print("\n")
print("=" * 100)
print("1단계 학습")
print("=" * 100)

print("LOSS       : MSE")
print("BATCH SIZE : 32")


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)


start1 = time.time()


history1 = model.fit(
    x_train,
    y_train,

    epochs=30,

    batch_size=32,

    validation_split=0.1,

    callbacks=[
        es1
    ],

    verbose=1
)


end1 = time.time()


# -------------------------------------------------------------------------
# 1단계 평가
# -------------------------------------------------------------------------

loss1, acc1 = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n")
print("=" * 100)
print("1단계 결과")
print("=" * 100)

print("Loss     :", loss1)
print("Accuracy :", acc1)


# =================================================================================
# 9. 2단계
#
# Loss:
# MSE 그대로
#
# Batch:
# 32 → 512
#
#
# ★ 중요
#
# 여기에서는 compile을 다시 하지 않는다.
#
# 이유:
#
# 1단계의
#
# - Weight
# - Bias
# - Adam optimizer 상태
#
# 를 그대로 유지하면서
#
# Batch Size만
#
# 32 → 512
#
# 로 변경하기 위해서이다.
#
#
# model.fit()의 batch_size는
# compile과 관계없기 때문에
#
# 그냥 batch_size만 바꾸면 된다.
# =================================================================================

print("\n")
print("=" * 100)
print("2단계 학습")
print("=" * 100)

print("LOSS       : MSE")
print("BATCH SIZE : 512")


start2 = time.time()


history2 = model.fit(
    x_train,
    y_train,

    epochs=100,

    batch_size=512,

    validation_split=0.1,

    callbacks=[
        es2
    ],

    verbose=1
)


end2 = time.time()


# -------------------------------------------------------------------------
# 2단계 평가
# -------------------------------------------------------------------------

loss2, acc2 = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n")
print("=" * 100)
print("2단계 결과")
print("=" * 100)

print("Loss     :", loss2)
print("Accuracy :", acc2)


# =================================================================================
# 10. 3단계
#
# Loss 변경
#
# MSE
#  ↓
# Categorical Crossentropy
#
#
# Batch:
# 512 그대로
#
#
# 여기서는 Loss를 바꿔야 하기 때문에
# compile이 필요하다.
#
#
# compile을 다시 해도
#
# Conv / Dense의
#
# Weight와 Bias는 유지된다.
#
#
# 단,
#
# 새로운 Adam optimizer가 생성되므로
# optimizer 내부 상태는 초기화된다.
# =================================================================================

print("\n")
print("=" * 100)
print("3단계 학습")
print("=" * 100)

print("LOSS       : Categorical Crossentropy")
print("BATCH SIZE : 512")


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


start3 = time.time()


history3 = model.fit(
    x_train,
    y_train,

    epochs=100,

    batch_size=512,

    validation_split=0.1,

    callbacks=[
        es3
    ],

    verbose=1
)


end3 = time.time()


# =================================================================================
# 11. 최종 평가
# =================================================================================

loss3, acc3 = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


print("\n")
print("=" * 100)
print("최종 결과")
print("=" * 100)

print("1단계 MSE / Batch 32  :", acc1)
print("2단계 MSE / Batch 512 :", acc2)
print("3단계 CE  / Batch 512 :", acc3)


# =================================================================================
# 12. Predict
# =================================================================================

y_pred = model.predict(
    x_test,
    batch_size=512,
    verbose=0
)


print("\nPrediction Shape :", y_pred.shape)

# (10000, 100)


# =================================================================================
# Softmax
#
# 예:
#
# [0.01, 0.02, ..., 0.80, ..., 0.01]
#
#                  ↓
#
# 가장 큰 확률을 가진 index
#
#                  ↓
#
# 클래스 번호
# =================================================================================

y_pred_class = np.argmax(
    y_pred,
    axis=1
)


# =================================================================================
# 실제 정답
#
# (10000, 1)
#      ↓
# (10000,)
# =================================================================================

y_true = y_test_original.reshape(-1)


# =================================================================================
# sklearn Accuracy
# =================================================================================

accuracy = accuracy_score(
    y_true,
    y_pred_class
)


print("\n")
print("=" * 100)
print("sklearn Accuracy")
print("=" * 100)

print(accuracy)


# =================================================================================
# 13. Training Time
# =================================================================================

time1 = end1 - start1
time2 = end2 - start2
time3 = end3 - start3


print("\n")
print("=" * 100)
print("Training Time")
print("=" * 100)

print(
    "1단계 MSE / Batch 32  :",
    round(time1, 2),
    "초"
)

print(
    "2단계 MSE / Batch 512 :",
    round(time2, 2),
    "초"
)

print(
    "3단계 CE  / Batch 512 :",
    round(time3, 2),
    "초"
)

print(
    "Total                 :",
    round(
        time1 + time2 + time3,
        2
    ),
    "초"
)

# 최종 스코어 0.3235