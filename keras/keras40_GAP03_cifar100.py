# 39 - Stacking Ensemble Memory Optimized (CIFAR-100)

import numpy as np
import tensorflow as tf
import time
import os
import gc

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# =================================================================================
# 0. Seed & 변수 초기화
# =================================================================================
np.random.seed(333)
tf.random.set_seed(333)

EPOCHS_SMALL = 30
BATCH_SMALL = 32
EPOCHS_LARGE = 100
BATCH_LARGE = 512

save_dir = './_data/CNN/'
os.makedirs(save_dir, exist_ok=True)
model_paths = [os.path.join(save_dir, f'keras39_cifar100_base{i}.h5') for i in range(1, 6)]

es1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
es2 = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# =================================================================================
# 1. 데이터
# =================================================================================
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 2. X Scaling
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_test = (x_test.astype(np.float32) - 127.5) / 127.5

# 3. Y OneHotEncoding
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

x_train_base, x_train_meta, y_train_base, y_train_meta = train_test_split(
    x_train, y_train, train_size=0.7, random_state=42
)

print(f"Base Training Data: {x_train_base.shape}")
print(f"Meta Training Data: {x_train_meta.shape}")
print(f"Test Data: {x_test.shape}")

# =================================================================================
# 2. Base 모델 구성 함수
# =================================================================================
def create_model_1():
    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(100, activation='softmax')
    ], name="BaseModel_1_StandardCNN")
    return model

def create_model_2():
    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(100, activation='softmax')
    ], name="BaseModel_2_DeepCNN")
    return model

def create_model_3():
    model = Sequential([
        Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(100, activation='softmax')
    ], name="BaseModel_3_LargeKernelCNN")
    return model

def create_model_4():
    model = Sequential([
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(100, activation='softmax')
    ], name="BaseModel_4_WideShallowCNN")
    return model

def create_model_5():
    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        AveragePooling2D(),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        AveragePooling2D(),
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(100, activation='softmax')
    ], name="BaseModel_5_AvgPoolCNN")
    return model

model_creators = [create_model_1, create_model_2, create_model_3, create_model_4, create_model_5]

# =================================================================================
# 3. Base 모델 개별 훈련 및 해제 (OOM 방지)
# =================================================================================
def train_and_save_base_model(model_fn, x, y, model_path, model_num):
    if os.path.exists(model_path):
        print(f"--- Base Model {model_num} Already Exists. Skipping Training. ---")
        return

    print(f"\n--- Training Base Model {model_num} ---")
    model = model_fn()
    
    print(f"[Model {model_num}] 1단계 학습 (MSE, Batch {BATCH_SMALL})")
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x, y, epochs=EPOCHS_SMALL, batch_size=BATCH_SMALL, validation_split=0.1, callbacks=[es1], verbose=1)
    
    print(f"[Model {model_num}] 2단계 학습 (CE, Batch {BATCH_LARGE})")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=EPOCHS_LARGE, batch_size=BATCH_LARGE, validation_split=0.1, callbacks=[es2], verbose=1)
    
    model.save(model_path)
    print(f"--- Base Model {model_num} Saved to {model_path} ---")
    
    # 훈련 끝난 후 메모리 해제
    del model
    K.clear_session()
    gc.collect()

# 순차적으로 모델 훈련 진행
for i, (creator, path) in enumerate(zip(model_creators, model_paths)):
    train_and_save_base_model(creator, x_train_base, y_train_base, path, i+1)

# =================================================================================
# 4. Meta Feature 생성 (순차적 예측으로 OOM 방지)
# =================================================================================
print("\n--- Generating Meta Features ---")
meta_train_list = []
meta_test_list = []

for i, path in enumerate(model_paths):
    print(f"Loading Base Model {i+1} for prediction...")
    model = load_model(path)
    
    pred_meta = model.predict(x_train_meta, batch_size=BATCH_LARGE, verbose=0)
    pred_test = model.predict(x_test, batch_size=BATCH_LARGE, verbose=0)
    
    meta_train_list.append(pred_meta)
    meta_test_list.append(pred_test)
    
    # 예측이 끝난 모델 메모리 해제
    del model
    K.clear_session()
    gc.collect()

# 리스트에 모인 5개 모델의 예측값을 옆으로 이어붙임
x_meta_train = np.concatenate(meta_train_list, axis=1)
x_meta_test = np.concatenate(meta_test_list, axis=1)

print(f"Meta Feature Train Shape: {x_meta_train.shape}")
print(f"Meta Feature Test Shape: {x_meta_test.shape}")

# =================================================================================
# 5. Meta Model 훈련
# =================================================================================
print("\n--- Training Meta Model ---")
meta_model = Sequential([
    Dense(1024, activation='relu', input_shape=(500,)),  # 5 models * 100 classes
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(100, activation='softmax')
], name="MetaModel")

meta_es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("[Meta Model] 1단계 학습 (MSE)")
meta_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
meta_model.fit(x_meta_train, y_train_meta, epochs=EPOCHS_SMALL, batch_size=BATCH_SMALL, validation_split=0.1, callbacks=[meta_es], verbose=1)

print("[Meta Model] 2단계 학습 (CE)")
meta_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
meta_model.fit(x_meta_train, y_train_meta, epochs=EPOCHS_LARGE, batch_size=BATCH_LARGE, validation_split=0.1, callbacks=[meta_es], verbose=1)

# =================================================================================
# 6. 평가, 예측
# =================================================================================
print("\n=================== Model Evaluations ===================")
meta_loss = meta_model.evaluate(x_meta_test, y_test, verbose=0)
print(f"--- Meta Model (Ensemble) Accuracy: {meta_loss[1]:.4f} ---")

y_pred_meta = meta_model.predict(x_meta_test, verbose=0)
y_pred_meta_arg = np.argmax(y_pred_meta, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_pred_meta_arg)
print(f"Final Stacking Ensemble Accuracy_score: {acc_score:.4f}")

# 1차 0.3444