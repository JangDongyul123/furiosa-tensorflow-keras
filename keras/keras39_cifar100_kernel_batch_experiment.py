import numpy as np
import tensorflow as tf
import time
import gc

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K

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
# 2. 실험 파라미터 설정
# =================================================================================
# 테스트해볼 커널 사이즈와 배치 사이즈 리스트
kernel_sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
batch_sizes = [64, 256, 1024]
MAX_EPOCHS = 50

# 얼리스탑 설정 (너무 오래 돌지 않도록 patience를 작게 설정)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

results = []

# =================================================================================
# 3. 실험 시작 (이중 루프)
# =================================================================================
for k_size in kernel_sizes:
    for b_size in batch_sizes:
        print("\n" + "="*60)
        print(f"▶▶▶ 실험 진행 중: Kernel Size = {k_size}, Batch Size = {b_size}")
        print("="*60)
        
        # 모델 구성 (간결한 VGG 형태의 표준 CNN 모델 사용)
        model = Sequential([
            Conv2D(64, kernel_size=k_size, padding='same', activation='relu', input_shape=(32, 32, 3)),
            Conv2D(64, kernel_size=k_size, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(128, kernel_size=k_size, padding='same', activation='relu'),
            Conv2D(128, kernel_size=k_size, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(100, activation='softmax')
        ], name=f"CNN_K{k_size[0]}_B{b_size}")
        
        # 컴파일 및 훈련 (여기선 빠른 실험을 위해 CE 단일 로스만 사용)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        # verbose=1로 두면 너무 길어질 수 있어 상황에 따라 verbose=2나 0으로 줄일 수 있습니다.
        model.fit(x_train, y_train, epochs=MAX_EPOCHS, batch_size=b_size, 
                  validation_split=0.2, callbacks=[es], verbose=1)
        end_time = time.time()
        
        # 평가
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        time_taken = round(end_time - start_time, 2)
        
        print(f"👉 [결과] Accuracy: {acc:.4f} / Time: {time_taken}s")
        
        # 결과 저장
        results.append({
            'Kernel Size': k_size,
            'Batch Size': b_size,
            'Time': time_taken,
            'Accuracy': acc
        })
        
        # OOM 방지: 루프가 한 번 끝날 때마다 모델 메모리 강제 삭제
        del model
        K.clear_session()
        gc.collect()

# =================================================================================
# 4. 결과 요약 리포트
# =================================================================================
print("\n\n" + "="*60)
print("                  최종 실험 결과 요약 (CIFAR-100)")
print("="*60)
print(f"{'Kernel Size':<15} | {'Batch Size':<15} | {'Time (s)':<15} | {'Accuracy':<15}")
print("-" * 65)

for res in results:
    k_str = str(res['Kernel Size'])
    b_str = str(res['Batch Size'])
    t_str = str(res['Time'])
    a_str = f"{res['Accuracy']:.4f}"
    print(f"{k_str:<15} | {b_str:<15} | {t_str:<15} | {a_str:<15}")

print("="*60)

# 가장 성능이 좋은 조합 출력
best_res = max(results, key=lambda x: x['Accuracy'])
print(f"\n🌟 최고 성능 조합: Kernel {best_res['Kernel Size']}, Batch {best_res['Batch Size']} -> Accuracy: {best_res['Accuracy']:.4f}\n")

