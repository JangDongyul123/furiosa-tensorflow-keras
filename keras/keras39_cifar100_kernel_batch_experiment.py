import numpy as np
import tensorflow as tf
import time
import gc

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
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
batch_sizes = [16, 32, 64]
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
            
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(100, activation='softmax')
        ], name=f"CNN_K{k_size[0]}_B{b_size}")
        
        # 컴파일 및 훈련 (여기선 빠른 실험을 위해 CE 단일 로스만 사용)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        # verbose=1로 두면 너무 길어질 수 있어 상황에 따라 verbose=2나 0으로 줄일 수 있습니다.
        model.fit(x_train, y_train, epochs=MAX_EPOCHS, batch_size=b_size, 
                  validation_split=0.2, callbacks=[es], verbose=0)
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
# 4. 결과 요약 리포트 및 랭킹 (Accuracy 기준 내림차순)
# =================================================================================
print("\n\n" + "="*75)
print("              최종 실험 랭킹 (CIFAR-100, Accuracy 기준)")
print("="*75)
print(f"{'Rank':<5} | {'Kernel Size':<15} | {'Batch Size':<12} | {'Time (s)':<10} | {'Accuracy':<10}")
print("-" * 75)

# 결과를 Accuracy 기준으로 내림차순 정렬
results.sort(key=lambda x: x['Accuracy'], reverse=True)

for i, res in enumerate(results):
    rank_str = f"{i + 1}위"
    k_str = str(res['Kernel Size'])
    b_str = str(res['Batch Size'])
    t_str = str(res['Time'])
    a_str = f"{res['Accuracy']:.4f}"
    print(f"{rank_str:<5} | {k_str:<15} | {b_str:<12} | {t_str:<10} | {a_str:<10}")

print("="*75)

# 가장 성능이 좋은 조합 출력
best_res = results[0]
print(f"\n🌟 1위 최고 성능 조합: Kernel {best_res['Kernel Size']}, Batch {best_res['Batch Size']} -> Accuracy: {best_res['Accuracy']:.4f}\n")

# ===========================================================================        
#               Flatten 사용 1차 실험 랭킹 (CIFAR-100, Accuracy 기준)
# ===========================================================================        
# Rank  | Kernel Size     | Batch Size   | Time (s)   | Accuracy
# ---------------------------------------------------------------------------        
# 1위    | (2, 2)          | 64           | 48.84      | 0.4153
# 2위    | (3, 3)          | 64           | 44.32      | 0.4104
# 3위    | (3, 3)          | 256          | 45.08      | 0.4034
# 4위    | (2, 2)          | 256          | 55.72      | 0.3996
# 5위    | (4, 4)          | 256          | 87.18      | 0.3965
# 6위    | (3, 3)          | 1024         | 71.87      | 0.3901
# 7위    | (4, 4)          | 1024         | 127.37     | 0.3888
# 8위    | (2, 2)          | 1024         | 78.62      | 0.3832
# 9위    | (5, 5)          | 256          | 74.59      | 0.3729
# 10위   | (4, 4)          | 64           | 90.17      | 0.3683
# 11위   | (5, 5)          | 1024         | 95.14      | 0.3614
# 12위   | (5, 5)          | 64           | 67.76      | 0.3420
# ===========================================================================        

# 🌟 1위 최고 성능 조합: Kernel (2, 2), Batch 64 -> Accuracy: 0.4153


# ===========================================================================        
#               Flatten 사용 2차 실험 랭킹 (CIFAR-100, Accuracy 기준)
# ===========================================================================        
# Rank  | Kernel Size     | Batch Size   | Time (s)   | Accuracy
# ---------------------------------------------------------------------------        
# 1위    | (3, 3)          | 64           | 48.6       | 0.4149
# 2위    | (2, 2)          | 32           | 53.76      | 0.4077
# 3위    | (2, 2)          | 64           | 46.73      | 0.4064
# 4위    | (2, 2)          | 16           | 86.37      | 0.3887
# 5위    | (3, 3)          | 32           | 46.7       | 0.3863
# 6위    | (4, 4)          | 64           | 82.22      | 0.3710
# 7위    | (3, 3)          | 16           | 73.33      | 0.3587
# 8위    | (4, 4)          | 32           | 98.95      | 0.3585
# 9위    | (5, 5)          | 64           | 67.6       | 0.3297
# 10위   | (4, 4)          | 16           | 120.26     | 0.3177
# 11위   | (5, 5)          | 32           | 85.67      | 0.3100
# 12위   | (5, 5)          | 16           | 104.7      | 0.2711
# ===========================================================================        

# 🌟 1위 최고 성능 조합: Kernel (3, 3), Batch 64 -> Accuracy: 0.4149

# ===========================================================================        
#               MaxPooling2D 실험 랭킹 (CIFAR-100, Accuracy 기준)
# ===========================================================================        
# Rank  | Kernel Size     | Batch Size   | Time (s)   | Accuracy
# ---------------------------------------------------------------------------        
# 1위    | (3, 3)          | 64           | 101.99     | 0.4874
# 2위    | (3, 3)          | 32           | 101.27     | 0.4643
# 3위    | (2, 2)          | 64           | 167.81     | 0.4630
# 4위    | (2, 2)          | 32           | 168.39     | 0.4469
# 5위    | (2, 2)          | 16           | 160.67     | 0.4390
# 6위    | (4, 4)          | 64           | 152.08     | 0.4350
# 7위    | (4, 4)          | 32           | 162.6      | 0.4346
# 8위    | (3, 3)          | 16           | 135.32     | 0.4234
# 9위    | (5, 5)          | 64           | 137.37     | 0.4177
# 10위   | (4, 4)          | 16           | 182.31     | 0.3977
# 11위   | (5, 5)          | 32           | 133.59     | 0.3670
# 12위   | (5, 5)          | 16           | 137.66     | 0.3357
# ===========================================================================        

# 🌟 1위 최고 성능 조합: Kernel (3, 3), Batch 64 -> Accuracy: 0.4874