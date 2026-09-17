# CNN Kernel Size & Batch Size Comparison (CIFAR-10)

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time

# =================================================================================
# 1. 데이터
# =================================================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 스케일링 (-1.0 ~ 1.0)
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

# =================================================================================
# 2. 모델 구성 함수 (커널 사이즈를 인자로 받음)
# =================================================================================
def create_model(k_size):
    model = Sequential([
        Conv2D(32, kernel_size=k_size, padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, kernel_size=k_size, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(64, kernel_size=k_size, padding='same', activation='relu'),
        Conv2D(64, kernel_size=k_size, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ], name=f"CNN_Kernel_{k_size[0]}x{k_size[1]}")
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# =================================================================================
# 3. 비교 실험 설정
# =================================================================================
kernel_sizes = [(3, 3), (5, 5)]  # 비교할 커널 사이즈 리스트
batch_sizes = [64, 256]          # 비교할 배치 사이즈 리스트
epochs = 15                      # 테스트를 위해 적절한 에포크 설정

results = []
# EarlyStopping: val_loss가 3번 연속 개선되지 않으면 멈춤
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)

# =================================================================================
# 4. 훈련 및 평가 루프
# =================================================================================
for k_size in kernel_sizes:
    for b_size in batch_sizes:
        print(f"\n=======================================================")
        print(f"▶ Training with Kernel Size: {k_size} | Batch Size: {b_size}")
        print(f"=======================================================")
        
        model = create_model(k_size)
        
        start_time = time.time()
        # 훈련 데이터의 20%를 검증용(validation)으로 사용
        model.fit(x_train, y_train, epochs=epochs, batch_size=b_size, 
                  validation_split=0.2, verbose=1, callbacks=[es])
        end_time = time.time()
        
        # 평가
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        
        # 결과 저장
        results.append({
            'Kernel_Size': f"{k_size[0]}x{k_size[1]}",
            'Batch_Size': b_size,
            'Loss': round(loss, 4),
            'Accuracy': round(acc, 4),
            'Time(s)': round(end_time - start_time, 2)
        })

# =================================================================================
# 5. 결과 출력
# =================================================================================
df_results = pd.DataFrame(results)
print("\n======================= 최종 비교 결과 =======================")
print(df_results.to_markdown(index=False))
print("==============================================================")
