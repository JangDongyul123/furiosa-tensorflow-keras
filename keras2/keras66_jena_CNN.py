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
y_degrees = df['wd (deg)'].astype(np.float32).values[-total_len:]
y_radians = np.deg2rad(y_degrees)
y_all = np.column_stack((np.sin(y_radians), np.cos(y_radians))).astype(np.float32)

# x_all.shape = (420480, 13)
# y_all.shape = (420480, 2)  # 각 timestep의 sin/cos 풍향

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
y_blocks = y_all.reshape(-1, size, 2)
y_degree_blocks = y_degrees.reshape(-1, size, 1)

# x_blocks.shape = (2920, 144, 13)
# batch/block = 2920
# timestamp = 144
# feature = 13
#
# y_blocks.shape = (2920, 144, 2)
# batch/block = 2920
# timestamp = 144
# target features = sin(wd), cos(wd)

print("전체 데이터를 144개씩 묶은 총 상자(덩어리) 수:", x_blocks.shape[0])

# 3단계: 파이썬 인덱싱으로 문제와 정답 엇갈리기
# 파이썬에서 [-1]은 마지막 요소를 뜻합니다.

# [예측용 데이터 빼놓기]
y_predict_true = y_degree_blocks[-1:]  # 마지막 Block: 각도 단위의 최종 실제 정답
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

# 4단계: 시간 순서를 유지해 Train / Validation 분리
# 미래 날짜가 훈련 데이터에 섞이지 않도록 앞 80%를 훈련, 뒤 20%를 검증으로 둔다.
split_idx = int(len(x_train_full) * 0.8)
x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

# 시계열 데이터이므로 shuffle 기반 분할 대신 시간 순서를 유지한다.

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

# 풍향 target은 sin/cos로 변환되어 [-1, 1] 범위이므로 별도 MinMaxScaler는 사용하지 않는다.
# y_predict_true에는 평가용 원본 각도(degree)를 보존한다.

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

model_files = glob.glob(os.path.join(save_path, 'jena_wd_conv1d_*.keras'))

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

if len(model_files) > 0:
    # 이 Conv1D 실험이 저장한 모델만 불러와 다른 구조의 모델을 잘못 읽지 않게 합니다.
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

    model.add(Dense(2))
    # 각 timestamp마다 풍향의 sin/cos 값 2개를 출력
    # 출력 : (N, 144, 2)

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
# ↓ Dense(2)  # 풍향을 sin/cos 두 값으로 예측
#
# (N, 144, 2)
#
# ★ Conv1D에서 filters는 출력 feature(channel)의 개수가 된다.
# ★ kernel_size는 한 번에 몇 개의 연속된 timestep을 볼 것인지를 의미한다.

# ==========================================
# 3. 훈련
# ==========================================

# val_loss가 좋아질 때마다 모델 저장
import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")
mcp_filename = 'jena_wd_conv1d_' + date + '_{epoch:04d}-{val_loss:.4f}.keras'
filepath = os.path.join(save_path, mcp_filename)

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

# 훈련에 사용하지 않은 마지막 하루 데이터(x_predict)를 입력해 다음 하루의 풍향(sin/cos)을 예측
result = model.predict(x_predict)

# result.shape = (1, 144, 2)

predicted_degrees = np.mod(
    np.degrees(np.arctan2(result[0, :, 0], result[0, :, 1])),
    360.0,
)
actual_degrees = y_predict_true[0, :, 0]

# 원형 오차를 -180~180 범위로 계산해 359도와 1도의 차이를 2도로 처리한다.
angular_error = (predicted_degrees - actual_degrees + 180.0) % 360.0 - 180.0
rmse = np.sqrt(np.mean(np.square(angular_error)))

print('예측결과 shape : ', predicted_degrees.shape)  # (144,)
print('첫 5개 예측값 : \n', predicted_degrees[:5])
print('첫 5개 실제값 : \n', y_predict_true[0, :5, 0])
print('풍향 원형 RMSE (degree) :', rmse)

# ==========================================
# 5. 결과를 CSV 파일로 저장하기
# ==========================================
# 예측값과 실제 정답을 비교할 수 있도록 DataFrame으로 만듭니다.
submit_df = pd.DataFrame({
    'Actual_wd': y_predict_true[0, :, 0],  # 실제 정답 144개
    'Predicted_wd': predicted_degrees     # 예측값 144개
})

csv_path = save_path + 'jena_predict_result.csv'
submit_df.to_csv(
    csv_path,
    index=False,
    encoding='utf-8'
)

print(f"\n[알림] 144개의 예측 결과가 CSV 파일로 성공적으로 저장되었습니다!\n▶ 저장 위치: {csv_path}")
