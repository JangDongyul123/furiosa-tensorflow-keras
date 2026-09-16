# 36 - Stacking Ensemble (MNIST)

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, AveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =================================================================================
# 1. 데이터
# =================================================================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 스케일링 (Max ABS)
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# OneHotEncoding
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

x_train_base, x_train_meta, y_train_base, y_train_meta = train_test_split(
    x_train, y_train, train_size=0.8, random_state=42
)

print(f"Base Training Data: {x_train_base.shape}")
print(f"Meta Training Data: {x_train_meta.shape}")
print(f"Test Data: {x_test.shape}")

# =================================================================================
# 2. Base 모델 구성
# =================================================================================

def create_model_1():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(16, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ], name="BaseModel_1_StandardCNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_2():
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ], name="BaseModel_2_DeepCNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_3():
    model = Sequential([
        Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ], name="BaseModel_3_LargeKernelCNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_4():
    # 4. 얕지만 필터 수가 많은 CNN
    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ], name="BaseModel_4_WideShallowCNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_5():
    # 5. AveragePooling 을 사용하는 CNN
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        AveragePooling2D(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        AveragePooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ], name="BaseModel_5_AvgPoolCNN")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model1 = create_model_1()
model2 = create_model_2()
model3 = create_model_3()
model4 = create_model_4()
model5 = create_model_5()

# =================================================================================
# 3. Base 모델 훈련
# =================================================================================
EPOCHS_BASE = 30
BATCH_SIZE = 128

print("--- Training Base Model 1 ---")
model1.fit(x_train_base, y_train_base, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2)

print("--- Training Base Model 2 ---")
model2.fit(x_train_base, y_train_base, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2)

print("--- Training Base Model 3 ---")
model3.fit(x_train_base, y_train_base, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2)

print("--- Training Base Model 4 ---")
model4.fit(x_train_base, y_train_base, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2)

print("--- Training Base Model 5 ---")
model5.fit(x_train_base, y_train_base, epochs=EPOCHS_BASE, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2)

# =================================================================================
# 4. Meta Feature 생성 (Prediction)
# =================================================================================
print("--- Generating Meta Features ---")
pred1_meta = model1.predict(x_train_meta)
pred2_meta = model2.predict(x_train_meta)
pred3_meta = model3.predict(x_train_meta)
pred4_meta = model4.predict(x_train_meta)
pred5_meta = model5.predict(x_train_meta)

x_meta_train = np.concatenate([pred1_meta, pred2_meta, pred3_meta, pred4_meta, pred5_meta], axis=1)

pred1_test = model1.predict(x_test)
pred2_test = model2.predict(x_test)
pred3_test = model3.predict(x_test)
pred4_test = model4.predict(x_test)
pred5_test = model5.predict(x_test)

x_meta_test = np.concatenate([pred1_test, pred2_test, pred3_test, pred4_test, pred5_test], axis=1)

# =================================================================================
# 5. Meta Model 훈련
# =================================================================================
print("--- Training Meta Model ---")
meta_model = Sequential([
    Dense(128, activation='relu', input_shape=(50,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
], name="MetaModel")

meta_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Meta 모델 학습
meta_model.fit(x_meta_train, y_train_meta, epochs=50, batch_size=128, verbose=1, validation_split=0.2)

# =================================================================================
# 6. 평가, 예측
# =================================================================================
print("=================== Model Evaluations ===================")
# 개별 모델 평가
loss1 = model1.evaluate(x_test, y_test, verbose=0)
loss2 = model2.evaluate(x_test, y_test, verbose=0)
loss3 = model3.evaluate(x_test, y_test, verbose=0)
loss4 = model4.evaluate(x_test, y_test, verbose=0)
loss5 = model5.evaluate(x_test, y_test, verbose=0)
meta_loss = meta_model.evaluate(x_meta_test, y_test, verbose=0)

print(f"Base Model 1 Accuracy: {loss1[1]:.4f}")
print(f"Base Model 2 Accuracy: {loss2[1]:.4f}")
print(f"Base Model 3 Accuracy: {loss3[1]:.4f}")
print(f"Base Model 4 Accuracy: {loss4[1]:.4f}")
print(f"Base Model 5 Accuracy: {loss5[1]:.4f}")
print(f"--- Meta Model (Ensemble) Accuracy: {meta_loss[1]:.4f} ---")

# 최종 예측값
y_pred_meta = meta_model.predict(x_meta_test)
y_pred_meta_arg = np.argmax(y_pred_meta, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_pred_meta_arg)
print(f"Final Stacking Ensemble Accuracy_score: {acc_score:.4f}")