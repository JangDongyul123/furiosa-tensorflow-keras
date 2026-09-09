# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

# path = 'c:/study/_data/santander/' # 절대 경로
path = './_data/kaggle_santander/' # 상대 경로

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
   
print(train_csv.shape) # (200000, 201) 
print(test_csv.shape) # (200000, 200) 
print(submission.shape) # (200000, 1) 

print(train_csv.info())
# 데이터가 너무 많아서 잘 안됨

print(train_csv.isna().sum()) # 결측치 없음
print(test_csv.isnull().sum()) # 통계치 확인

x = train_csv.drop(['target'], axis=1) # target 컬럼 제거
y = train_csv['target']
print(x.shape, y.shape) #(200000, 200) (200000, )

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([179902,  20098], dtype=int64))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.5, random_state=333, stratify=y
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=x.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
# 이진 분류의 핵심: 활성화 함수 sigmoid
# 모델이 어떤 값을 출력하든, sigmoid 함수를 거치면 무조건 0.0 ~ 1.0 사이의 확률값(예: 0.82)으로 변환해 줍니다.

# 3. 컴파일, 훈련
# compile: 모델이 데이터를 어떻게 학습할지 '학습 방법'을 설정합니다.
# - loss (손실): 예측값과 실제 정답(0 또는 1) 간의 '오차'입니다. 값이 작을수록 모델이 똑똑해지고 있다는 뜻입니다. 
#                이진 분류에서는 오차 계산을 위해 무조건 'binary_crossentropy'를 사용합니다.
# - metrics (평가지표): 오차(loss) 외에 사람이 직관적으로 성능을 알 수 있게 '정확도(accuracy)'를 추가합니다. 
#                       전체 문제 중 정답을 맞춘 비율을 뜻하며 1.0에 가까울수록 좋습니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

es = EarlyStopping(
    monitor='val_loss', # 훈련 데이터가 아닌 검증용 데이터의 오차(val_loss)를 감시합니다.
    mode='min',         # 오차는 작아질수록(min) 좋으므로 최소화되는 방향으로 관찰합니다.
    patience=20,        # 20번의 epoch 동안 더 이상 오차가 줄어들지 않으면 과적합으로 판단하고 강제 종료합니다.
    restore_best_weights=True # 강제 종료 시, 가장 오차가 작았던(성능이 좋았던) 에포크 시점의 가중치로 되돌립니다.
)

start_time = time.time()
print("\n========== [로그] 모델 훈련(fit) 시작 ==========")
# fit: 실제 데이터로 가중치를 업데이트(학습)하는 과정입니다.
hist = model.fit(
    x_train, y_train, 
    epochs=100,           # 전체 데이터를 100번 반복 학습합니다.
    batch_size=10240,     # 한 번에 10240개씩 묶어서 학습시킵니다.
    verbose=1,            # 프로그레스바로 학습 경과를 출력합니다.
    validation_split=0.5, # 훈련 데이터의 50%를 모의고사(val)용으로 빼서 과적합을 체크합니다.
    callbacks=[es]        # 위에서 만든 조기종료(es) 규칙을 적용합니다.
)
print("========== [로그] 모델 훈련(fit) 종료 ==========\n")
end_time = time.time()

# 4. 평가, 예측
# evaluate: 한 번도 학습하지 않은 시험지(x_test, y_test)로 모델의 진짜 실력을 평가합니다.
loss = model.evaluate(x_test, y_test)
print("\n========== [로그] 평가(evaluate) 결과 ==========")
print("1. 최종 loss (오차) :", loss[0], "-> 낮을 수록 예측을 잘한다는 의미입니다.")
print("2. 최종 accuracy (정확도) :", round(loss[1], 4), "-> 전체 중 맞춘 비율(0.95 = 95%)입니다. 높을 수록 좋습니다.")

y_pred = model.predict(x_test)
print("\n[로그] 원본 예측값(sigmoid 통과 후):\n", y_pred[:3])
# sigmoid를 거쳐서 나온 y_pred 값은 0~1 사이의 실수 확률입니다. 
# 이를 정답(0 또는 1)과 비교하기 위해 0.5를 기준으로 반올림(np.round)하여 0 또는 1로 변환합니다.
y_pred = np.round(y_pred) 
print("[로그] 반올림(np.round) 후 예측값:\n", y_pred[:3])

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

# 그래프 그리기 (matplotlib 활용)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='train_loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='green', label='train_accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='orange', label='val_accuracy')
plt.title('Model Loss & Accuracy')
plt.ylabel('Loss & Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# 제출용 파일(submission.csv) 생성
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission['target'] = y_submit
submission.to_csv(path + 'submission.csv')
print("submission.csv 생성 완료")