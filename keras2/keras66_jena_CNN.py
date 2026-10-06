# CONV1D() 가 중요 키워드다.
# 일반적으로 CNN에서는 CONV2D를 썼지만
# 시계열 데이터에서는 CONV1D를 사용한다.
# 왜냐하면 필터가 1차원이기 떄문이다.
# 직접 작성해봐야 한다.
# 수업 필기: 시계열 데이터에서는 Conv1D를 사용하며 필터가 1차원이다.
# 보완: Conv1D는 커널이 시계열의 한 축(timestamp 축)을 따라 이동하며 특징을 추출한다.
#       입력은 일반적으로 (batch, timestamp, feature)의 3차원 형태이다.
#       Conv2D는 이미지처럼 (height, width) 두 공간축을 따라 커널이 이동할 때 주로 사용한다.

import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 데이터 불러오기 및 전처리
# ==========================================
path = 'c:/study/furiosa-tensorflow-keras/keras/_data/kaggle_jena/'
df = pd.read_csv(path + 'jena_climate_2009_2016.csv')

# 시간 순서대로 정렬된 데이터이므로, 'Date Time' 컬럼은 훈련에 필요 없어서 삭제합니다.
# 이 컬럼을 지우면 총 14개의 컬럼(피처 13개 + 타겟 1개)이 남습니다.
df = df.drop('Date Time', axis=1)

# [훈련, 예측 데이터 구성: 과거로 미래 예측하기]
# 전체 데이터를 144개씩 한 덩어리로 묶어 상자에 담습니다. (메모리 폭발 방지)
size = 144
# 보완: Jena 데이터가 10분 간격이므로 144 timestep = 144 × 10분 = 1,440분 = 24시간이다.
#       따라서 이 코드에서는 한 Block이 하루 분량에 해당한다.
# 보완: 144개씩 묶는 주된 목적은 "메모리 폭발 방지"라기보다
#       하루 단위의 시계열 입력/출력 구조를 만들기 위한 것으로 보는 것이 더 정확하다.

# 1단계: 자투리 버리기
# 제나 데이터는 총 420,551줄입니다. 144개씩 딱 맞게 나누면 71개가 남습니다.
# len(df) // size 는 나머지를 버리고 몫(2920)만 구합니다.
# 여기에 다시 144를 곱해 420,480이라는 딱 떨어지는 숫자를 만듭니다.
total_len = (len(df) // size) * size

# (전체 원본 데이터) 맨 뒤에서부터 420,480개만 가져옵니다. (가장 오래된 71줄은 버림)
# 딥러닝(GPU)은 float32 연산을 많이 사용하므로 판다스 기본 float64를 float32로 줄이면 메모리 사용량도 감소합니다.
x_all = df.drop('wd (deg)', axis=1).astype(np.float32).values[-total_len:]
y_all = df['wd (deg)'].astype(np.float32).values[-total_len:]

# x_all.shape = (420480, 13)
# y_all.shape = (420480,)

# 2단계: reshape(-1, 144, 13)
# 길다란 데이터를 144개씩 끊어서 여러 개의 3차원 Block으로 만듭니다.
#
# [reshape(-1) 초간단 예시]
# 데이터가 [1, 2, 3, 4, 5, 6] 이렇게 6개 있다고 가정합니다.
# reshape(-1, 2)를 사용하면 전체 6개를 2개씩 묶으므로 -1은 자동으로 3이 됩니다.
#
# 결과:
# [[1, 2],
#  [3, 4],
#  [5, 6]]
#
# 즉 -1은 나머지 차원을 기준으로 크기를 자동 계산한다.
#
# 우리 코드에서는 전체 420,480줄을 144개로 나누면 2,920이므로
# x는 (2920, 144, 13) 형태가 됩니다.
x_blocks = x_all.reshape(-1, 144, 13)
y_blocks = y_all.reshape(-1, 144, 1)

# x_blocks.shape = (2920, 144, 13)
# batch/block = 2920
# timestamp = 144
# feature = 13
#
# y_blocks.shape = (2920, 144, 1)
# batch/block = 2920
# timestamp = 144
# target feature = 1

print("전체 데이터를 144개씩 묶은 총 상자(덩어리) 수:", x_blocks.shape[0])

# 3단계: 파이썬 인덱싱으로 문제와 정답 엇갈리기
# 파이썬에서 [-1]은 마지막 요소를 뜻합니다.

# [예측용 데이터 빼놓기]
y_predict_true = y_blocks[-1:]  # 마지막 Block: 최종 실제 정답
x_predict = x_blocks[-2:-1]     # 마지막 정답 바로 전 Block: 최종 예측 입력

# [훈련용 데이터 엇갈려서 주기]
# AI에게 "오늘(x)을 보고 다음 날(y)을 맞춰봐!"라고 훈련시키기 위해 한 Block씩 shift합니다.
x_train_full = x_blocks[:-2]    # Block 1 ~ 2918
y_train_full = y_blocks[1:-1]   # Block 2 ~ 2919

# 예:
# x[0] = 1일차 → y[0] = 2일차
# x[1] = 2일차 → y[1] = 3일차
# ...
# x_predict = 2919일차 → y_predict_true = 2920일차

# 4단계: Train / Val 분리
# 위에서 만든 2918쌍을 8:2 비율로 훈련용과 검증용으로 나눕니다.
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42
)

# 수업 필기: train_test_split으로 8:2 Train / Validation 분리
# 보완: train_test_split()은 기본값이 shuffle=True이므로 시계열 데이터의 시간 순서를 섞는다.
#       일반적인 미래 예측 문제에서는 미래 데이터를 Training에 넣고 과거 데이터를 Validation에 넣는 상황이 생길 수 있다.
#       따라서 실제 시계열 검증에서는 시간 순서를 유지하는 방식이 더 적절하다.
#
# 예:
# split_idx = int(len(x_train_full) * 0.8)
# x_train = x_train_full[:split_idx]
# y_train = y_train_full[:split_idx]
# x_val = x_train_full[split_idx:]
# y_val = y_train_full[split_idx:]

# ==========================================
# ★ 1-2. 데이터 스케일링 (Data Leakage 방지)
# ==========================================

# 반드시 Train / Validation을 나눈 이후 Train 데이터만으로 scaler를 fit한다.
# Validation과 Predict 데이터에는 fit하지 않고 transform만 사용한다.

# Conv1D 입력은 (batch, timestamp, feature)의 3차원이지만
# sklearn의 MinMaxScaler는 기본적으로 2차원 입력을 받으므로 잠시 2차원으로 변경합니다.
x_train = x_train.reshape(-1, 13)
x_val = x_val.reshape(-1, 13)
x_predict = x_predict.reshape(-1, 13)

scaler_x = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)  # 기준은 오직 Train
x_val = scaler_x.transform(x_val)
x_predict = scaler_x.transform(x_predict)

# 스케일링 후 다시 Conv1D 입력 형태인 3차원으로 복구
x_train = x_train.reshape(-1, 144, 13)
x_val = x_val.reshape(-1, 144, 13)
x_predict = x_predict.reshape(-1, 144, 13)

# Y(정답) 데이터도 스케일링합니다.
# 수업 필기: Y도 기울기 폭발 방지를 위해 동일하게 스케일링한다.
# 보완: Y 스케일링은 단순히 "기울기 폭발 방지"만을 위한 것은 아니다.
#       Target 값의 범위를 줄이면 Loss와 Gradient의 수치 범위가 안정되어 최적화가 쉬워질 수 있다.
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)  # 기준은 오직 Train
y_val = scaler_y.transform(y_val)

y_train = y_train.reshape(-1, 144, 1)
y_val = y_val.reshape(-1, 144, 1)

# y_predict_true는 마지막에 실제 예측값과 비교할 원본 정답이므로 스케일링하지 않는다.

print("훈련 데이터 쉐이프:", x_train.shape, y_train.shape)
print("검증 데이터 쉐이프:", x_val.shape, y_val.shape)
print("예측 문제 쉐이프:", x_predict.shape)
print("예측 정답 쉐이프:", y_predict_true.shape)

# ==========================================
# ★ 풍향(wd)의 특징
# ==========================================
# 수업 코드에서는 wd(deg)를 0~360의 일반적인 연속형 숫자로 보고 MinMaxScaler + MSE를 사용한다.
#
# 보완: 풍향은 원형(Circular) 데이터라는 점을 주의해야 한다.
# 예를 들어 실제값 359도와 예측값 1도는 실제 방향 차이가 2도밖에 안 나지만
# 일반적인 MSE에서는 두 값을 358 차이로 판단한다.
#
# 따라서 실전에서는 풍향을 다음처럼 sin/cos 두 값으로 변환하는 방식이 더 자연스럽다.
#
# rad = np.deg2rad(degree)
# y_sin = np.sin(rad)
# y_cos = np.cos(rad)
#
# 모델이 sin, cos 두 값을 예측한 뒤:
#
# pred_rad = np.arctan2(pred_sin, pred_cos)
# pred_deg = np.rad2deg(pred_rad)
# pred_deg = np.mod(pred_deg + 360, 360)
#
# 시험이 현재 수업 코드 기준이라면 MinMaxScaler + MSE 구조도 함께 기억하되,
# 실제 풍향 예측에서는 Circular 처리 방식이 더 적절하다는 것을 구분해두는 것이 좋다.

# ==========================================
# 2. 모델 구성 (또는 불러오기)
# ==========================================
save_path = 'c:/study/furiosa-tensorflow-keras/keras/_save/keras58/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_files = glob.glob(save_path + '*.keras')

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

if len(model_files) > 0:
    # 저장된 모델 중 가장 최근 수정된 모델을 찾아서 불러옵니다.
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"\n[알림] 저장된 가장 최근 모델을 불러와서 이어서 훈련(Resume Training)합니다: {latest_model_path}")
    model = load_model(latest_model_path)
    model.summary()

else:
    print("\n[알림] 저장된 모델이 없습니다. 새로 모델을 구성하고 훈련을 시작합니다.")

    model = Sequential()
    model.add(Input(shape=(144, 13)))
    # Input의 shape에는 batch_size를 쓰지 않는다.
    # 실제 입력 전체 shape = (batch, 144, 13)
    # model Input shape = (144, 13)
    # 즉 input_shape는 전체 X shape보다 batch 축 하나가 적다.

    # CNN(Conv1D)을 사용하여 시계열 데이터를 처리합니다.
    model.add(Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    # 입력 : (N, 144, 13)
    # 출력 : (N, 144, 128)
    #
    # filters=128
    # → 서로 다른 특징을 찾는 Filter 128개
    # → 출력 feature/channel이 128개가 된다.
    #
    # kernel_size=3
    # → timestamp를 한 번에 3칸씩 보면서 특징을 추출한다.
    #
    # padding='same'
    # → stride=1일 때 timestamp 길이 144를 그대로 유지한다.

    model.add(Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    # 입력 : (N, 144, 128)
    # 출력 : (N, 144, 64)

    model.add(Dense(64, activation='relu'))
    # Dense는 마지막 차원에 적용된다.
    # 입력 : (N, 144, 64)
    # 출력 : (N, 144, 64)

    model.add(Dense(1))
    # 각 timestamp마다 값 1개씩 출력
    # 출력 : (N, 144, 1)

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    model.summary()

# ==========================================
# Conv1D 핵심 Shape
# ==========================================
#
# Input
# (batch, timestamp, feature)
# (N, 144, 13)
#
# ↓ Conv1D(filters=128)
#
# (N, 144, 128)
#
# ↓ Conv1D(filters=64)
#
# (N, 144, 64)
#
# ↓ Dense(64)
#
# (N, 144, 64)
#
# ↓ Dense(1)
#
# (N, 144, 1)
#
# ★ Conv1D에서 filters는 출력 feature(channel)의 개수가 된다.
# ★ kernel_size는 한 번에 몇 개의 연속된 timestep을 볼 것인지를 의미한다.

# ==========================================
# 3. 훈련
# ==========================================

# val_loss가 좋아질 때마다 모델 저장
import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")
mcp_filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = save_path + 'k58_' + date + '_' + mcp_filename

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

# ModelCheckpoint
# monitor='val_loss'     → val_loss 감시
# mode='min'             → 작을수록 좋은 값
# save_best_only=True    → 이전 최고 성능보다 좋아졌을 때만 저장
# filepath               → 모델 저장 위치
#
# 수업 필기에서 MCP라고 줄여 부를 수 있지만 정확한 클래스명은 ModelCheckpoint이다.

# 검증 오차(val_loss)가 10번 연속 개선되지 않으면 훈련 조기 종료
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# patience=10
# → val_loss가 10 Epoch 동안 개선되지 않으면 종료
#
# restore_best_weights=True
# → 마지막 Epoch의 가중치가 아니라 val_loss가 가장 좋았던 Epoch의 가중치로 복구

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[es, mcp]
)

# batch_size=32
# → 전체 데이터를 32개 Sample씩 나누어 한 번씩 Gradient를 업데이트한다.
# Conv1D에 들어가는 한 Batch의 형태는 대략:
# (32, 144, 13)

# ==========================================
# 4. 평가 및 예측
# ==========================================
loss = model.evaluate(x_val, y_val)
print('val loss :', loss)

# 훈련에 사용하지 않은 마지막 하루 데이터(x_predict)를 입력해 다음 하루를 예측
result = model.predict(x_predict)

# result.shape = (1, 144, 1)

# ★ 스케일링 복구 (Inverse Transform)
# 예측된 결과를 원래 Target 단위로 되돌립니다.
result = scaler_y.inverse_transform(
    result.reshape(-1, 1)
).reshape(1, 144, 1)

print('예측결과 shape : ', result.shape)  # (1, 144, 1)
print('첫 5개 예측값 : \n', result[0, :5, 0])
print('첫 5개 실제값 : \n', y_predict_true[0, :5, 0])

# ==========================================
# 5. 결과를 CSV 파일로 저장하기
# ==========================================
# 예측값과 실제 정답을 비교할 수 있도록 DataFrame으로 만듭니다.
submit_df = pd.DataFrame({
    'Actual_wd': y_predict_true[0, :, 0],  # 실제 정답 144개
    'Predicted_wd': result[0, :, 0]        # 예측값 144개
})

csv_path = save_path + 'jena_predict_result.csv'
submit_df.to_csv(
    csv_path,
    index=False,
    encoding='utf-8'
)

print(f"\n[알림] 144개의 예측 결과가 CSV 파일로 성공적으로 저장되었습니다!\n▶ 저장 위치: {csv_path}")