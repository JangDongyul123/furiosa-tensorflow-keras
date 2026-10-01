import os
import datetime

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_path = os.path.join(
    '.',
    '_data',
    'kaggle_jena',
    'jena_climate_2009_2016.csv'
)

df = pd.read_csv(data_path)
df = df.drop('Date Time', axis=1)

size = 144
total_len = (len(df) // size) * size

wd_rad = np.deg2rad(df['wd (deg)'])

df['wd_sin'] = np.sin(wd_rad)
df['wd_cos'] = np.cos(wd_rad)

x_all = (
    df.drop(['wd (deg)', 'wd_sin', 'wd_cos'], axis=1)
      .astype(np.float32)
      .values[-total_len:]
)

y_all = (
    df[['wd_sin', 'wd_cos']]
      .astype(np.float32)
      .values[-total_len:]
)

x_blocks = x_all.reshape(-1, 144, 13)
y_blocks = y_all.reshape(-1, 144, 2)

y_true_deg_all = df['wd (deg)'].values[-total_len:]
y_true_deg_blocks = y_true_deg_all.reshape(-1, 144, 1)
y_predict_true_deg = y_true_deg_blocks[-1:]

x_predict = x_blocks[-2:-1]

x_train_full = x_blocks[:-2]
y_train_full = y_blocks[1:-1]

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    shuffle=False
)

x_train = x_train.reshape(-1, 13)
x_val = x_val.reshape(-1, 13)
x_predict = x_predict.reshape(-1, 13)

scaler_x = MinMaxScaler()

x_train = scaler_x.fit_transform(x_train)
x_val = scaler_x.transform(x_val)
x_predict = scaler_x.transform(x_predict)

x_train = x_train.reshape(-1, 144, 13)
x_val = x_val.reshape(-1, 144, 13)
x_predict = x_predict.reshape(-1, 144, 13)

model = Sequential([
    Input(shape=(144, 13)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(144 * 2, activation='linear'),
    Reshape((144, 2))
])

model.compile(
    loss='mse',
    optimizer='adam'
)

save_path = os.path.join(
    'c:/study/furiosa-tensorflow-keras/keras',
    '_save',
    'keras58_v10_wind_dnn_relu_single'
)

os.makedirs(save_path, exist_ok=True)

date = datetime.datetime.now().strftime("%m%d_%H%M")

filepath = os.path.join(
    save_path,
    f'single_dnn_{date}.h5'
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    factor=0.5,
    verbose=1
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=256,
    callbacks=[es, mcp, rlr],
    verbose=1
)

loss = model.evaluate(
    x_val,
    y_val,
    verbose=0
)

print(
    '\n* 단일 모델 최종 val loss (sin/cos MSE) :',
    loss
)

result = model.predict(
    x_predict,
    verbose=0
)

pred_sin = result[0, :, 0]
pred_cos = result[0, :, 1]

pred_rad = np.arctan2(
    pred_sin,
    pred_cos
)

pred_deg = np.rad2deg(pred_rad)
pred_deg = np.mod(pred_deg + 360, 360)

true_deg = y_predict_true_deg[0, :, 0]

print(
    '\n최종 예측 각도(첫 5개) :\n',
    pred_deg[:5]
)

print(
    '실제 정답 각도(첫 5개) :\n',
    true_deg[:5]
)

abs_diff = np.abs(
    true_deg - pred_deg
)

angle_diff = np.minimum(
    abs_diff,
    360 - abs_diff
)

deg_mae = np.mean(angle_diff)
deg_mse = np.mean(angle_diff ** 2)
deg_rmse = np.sqrt(deg_mse)

print('\n' + '=' * 50)
print(' * 단일 모델 최종 각도(Degree) 기준 오차 평가 * ')
print('=' * 50)
print(f' - MAE  : {deg_mae:.2f} 도')
print(f' - MSE  : {deg_mse:.2f}')
print(f' - RMSE : {deg_rmse:.2f} 도')
print('=' * 50)

submit_df = pd.DataFrame({
    'Actual_wd': true_deg,
    'Predicted_wd': pred_deg
})

csv_path = os.path.join(
    save_path,
    'jena_wind_dnn_relu_single_result.csv'
)

submit_df.to_csv(
    csv_path,
    index=False,
    encoding='utf-8'
)

print(
    f"\n[알림] 단일 모델 예측 결과가 저장되었습니다!"
    f"\n▶ 위치: {csv_path}"
)