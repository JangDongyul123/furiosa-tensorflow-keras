import os
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1단계: 데이터 불러오기 및 정리
# ==========================================
path = 'c:/study/furiosa-tensorflow-keras/keras/_data/kaggle_jena/'
df = pd.read_csv(path + 'jena_climate_2009_2016.csv')

# 시간 정보를 먼저 추출해 둡니다 (나중에 겨울 데이터만 필터링하기 위함)
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
months = df['Date Time'].dt.month.values

df = df.drop('Date Time', axis=1)

wd_rad = df['wd (deg)'] * (np.pi / 180)
df['wd_sin'] = np.sin(wd_rad)
df['wd_cos'] = np.cos(wd_rad)
df = df.drop('wd (deg)', axis=1) 

# 선생님 지시사항 반영: 과거 온도 피처 제거 (14개 컬럼)
x_all = df.drop('T (degC)', axis=1).values.astype(np.float64)
y_all = df[['T (degC)']].values.astype(np.float64)

# ==========================================
# 2단계: 스케일링
# ==========================================
# 실전 평가에 쓸 마지막 288행(문제 144개 + 정답 144개)을 제외하고 스케일링 기준을 잡습니다.
scaler = MinMaxScaler()
scaler.fit(x_all[:-288]) 
x_all_scaled = scaler.transform(x_all) 


# ==========================================
# 3단계: 슬라이딩 윈도우 ("과거 144행 -> 미래 144행 한 번에 예측")
# ==========================================
# 진정한 미래 예측(Forecasting) 룰입니다. 컨닝 없이 순수 실력으로 24시간 뒤를 맞춥니다.
size = 144
num_samples = len(df) - (size * 2) + 1  # X용 144개, Y용 144개가 필요함

X_data = np.empty((num_samples, size, 14), dtype=np.float32)
Y_data = np.empty((num_samples, size, 1), dtype=np.float32)

for i in range(num_samples):
    X_data[i] = x_all_scaled[i : i + size]                 # 오늘까지의 과거 144개 (X)
    Y_data[i] = y_all[i + size : i + size * 2]             # 내일 24시간의 미래 144개 정답 (Y)

# ==========================================
# 4단계: 훈련/실전 데이터 쪼개기
# ==========================================
# 맨 마지막 윈도우 1개가 바로 최종 평가할 "진짜 실전 문제(미래 144개 예측)"입니다.
x_실전 = X_data[-1:]   # 가장 마지막 과거 144개
y_실전_정답 = Y_data[-1:] # 가장 마지막 미래 144개 (파일의 맨 끝 144행)

# 실전용 1개를 제외한 나머지 약 42만 개 데이터
x_나머지 = X_data[:-1]
y_나머지 = Y_data[:-1]

# 타겟(Y)의 시작 시점이 겨울(11, 12, 1, 2월)인 데이터만 골라냅니다.
target_months = months[size : size + num_samples]
target_months_나머지 = target_months[:-1]
winter_indices = np.where(np.isin(target_months_나머지, [11, 12, 1, 2]))[0]

x_나머지 = x_나머지[winter_indices]
y_나머지 = y_나머지[winter_indices]
print(f"겨울 데이터 필터링 완료! 전체 데이터 중 겨울철 데이터 {len(x_나머지)}개만 남겼습니다.")

# 시계열 데이터 누수를 막기 위해 shuffle=False 설정 (시간순 분할)
x_train, x_val, y_train, y_val = train_test_split(x_나머지, y_나머지, test_size=0.2, shuffle=False)

# ==========================================
# 5단계 & 6단계: 스마트 모델 로드 및 훈련 준비
# ==========================================
save_path = 'c:/study/furiosa-tensorflow-keras/keras/_save/keras58_v9_dnn_swish/'
os.makedirs(save_path, exist_ok=True)

import glob
saved_models = glob.glob(save_path + '*.h5')

if saved_models:
    latest_model_path = max(saved_models, key=os.path.getctime)
    print(f"\n스마트 모드: 기존 훈련된 모델을 이어서 학습합니다 ({latest_model_path})")
    print("   (주의: 아예 처음부터 다시 학습하려면 폴더 안의 h5 파일들을 삭제해 주세요!)")
    model = load_model(latest_model_path)
    
    # 이어서 학습할 때 새로운 학습률을 강제 적용하고 싶다면 아래를 통해 덮어씌웁니다.
    model.optimizer.learning_rate.assign(0.0001)
else:
    print("\n저장된 모델이 없습니다. 완전히 새로운 모델을 생성합니다.")
    model = Sequential([
        Flatten(input_shape=(144, 14)),                    
        Dense(512, activation='swish'),            
        Dense(256, activation='swish'),            
        Dense(144, activation='linear'), # 한 방에 144개의 온도를 뱉어냅니다!
        Reshape((144, 1))
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))

date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = save_path + f'DNN_Swish_Forecasting_{date}.h5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=5, factor=0.5, verbose=1)

print("\n과거 144개로 미래 144개 예측 모델 훈련 시작")
model.fit(x_train, y_train, validation_data=(x_val, y_val), 
          epochs=500, batch_size=25600, callbacks=[es, mcp, rlr], verbose=1)

# 훈련 종료 후 (가중치가 복원된 최고 성능의) 최종 모델 저장
final_model_name = save_path + f'DNN_Swish_Forecasting_final_{date}.h5'
model.save(final_model_name)
print(f'최종 모델 저장 완료: {final_model_name}')

# ==========================================
# 7단계: 실전 예측 및 결과(RMSE) 계산
# ==========================================
# 폴더 내에서 가장 최근에 저장된 모델을 자동으로 찾아옵니다. (훈련 코드를 주석 처리해도 알아서 가져옵니다)
import glob
saved_models = glob.glob(save_path + '*.h5')
if saved_models:
    # 생성 시간이 가장 늦은(가장 최근) 파일을 찾습니다.
    latest_model = max(saved_models, key=os.path.getctime)
    print(f'\n최근에 저장된 모델을 불러옵니다: {latest_model}')
    model = load_model(latest_model)
else:
    raise FileNotFoundError("저장된 모델 파일(.h5)을 찾을 수 없습니다. 먼저 훈련을 진행해 주세요.")

예측결과 = model.predict(x_실전, verbose=0) 

# 예측결과와 정답 모두 144개의 값을 가짐
pred_temp = 예측결과[0, :, 0]       
true_temp = y_실전_정답[0, :, 0]    

test_rmse = np.sqrt(np.mean((pred_temp - true_temp)**2))
print(f"\nRMSE: {test_rmse:.4f} 도")

result_df = pd.DataFrame({
    '실제온도(True)': true_temp,
    '예측온도(Pred)': pred_temp,
    '오차(Error)': np.abs(true_temp - pred_temp)
})

csv_name = save_path + f'result_Forecasting_{date}.csv'
result_df.to_csv(csv_name, index=False, encoding='utf-8-sig')
print(f'결과 엑셀 저장 완료: {csv_name}')
