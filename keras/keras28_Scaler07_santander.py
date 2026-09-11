from sklearn.discriminant_analysis import StandardScaler
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# =================================================================================
# 0. Seed
# =================================================================================

np.random.seed(333)
tf.random.set_seed(333)


# =================================================================================
# 1. 데이터
# =================================================================================

path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print("train :", train_csv.shape)       # (200000, 201)
print("test  :", test_csv.shape)        # (200000, 200)
print("submit:", submission.shape)      # (200000, 1)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))


# =================================================================================
# 2. train / test 분리
# =================================================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=333,
    stratify=y
)


# =================================================================================
# 3. train / validation 분리
# =================================================================================

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=333,
    stratify=y_train
)

print("x_train :", x_train.shape)
print("x_val   :", x_val.shape)
print("x_test  :", x_test.shape)


# =================================================================================
# 4. MinMaxScaler
#
# x_train 데이터만 이용해서 Min / Max 값을 학습한다.
#
# x_val, x_test, Kaggle test_csv는
# x_train에서 학습한 동일한 기준으로 transform만 수행한다.
#
# 즉:
#
# x_train     -> fit_transform
# x_val       -> transform
# x_test      -> transform
# test_csv    -> transform
#
# validation / test 데이터의 정보를 scaler가 미리 보는 것을 방지한다.
# =================================================================================

scaler = MinMaxScaler()

# scaler = StandardScaler()

# scaler = MaxAbsScaler()

# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train).astype('float32')
x_val = scaler.transform(x_val).astype('float32')
x_test = scaler.transform(x_test).astype('float32')

# Kaggle 실제 제출 데이터도
# 반드시 x_train에서 학습한 MinMaxScaler 기준으로 변환
test_scaled = scaler.transform(test_csv).astype('float32')


print()
print("=" * 80)
print("MinMax Scaling 확인")
print("=" * 80)

print("x_train min / max :", np.min(x_train), np.max(x_train))
print("x_val   min / max :", np.min(x_val), np.max(x_val))
print("x_test  min / max :", np.min(x_test), np.max(x_test))
print("kaggle  min / max :", np.min(test_scaled), np.max(test_scaled))


# =================================================================================
# 4-1. 원-핫 인코딩 (One-Hot Encoding)
# =================================================================================

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print()
print("y_train (one-hot):", y_train.shape)
print("y_val   (one-hot):", y_val.shape)
print("y_test  (one-hot):", y_test.shape)


# =================================================================================
# 5. 모델
#
# 다중분류 구조
#
# 200
# ↓
# 128 ReLU
# ↓
# 64 ReLU
# ↓
# 32 ReLU
# ↓
# 2 Softmax (클래스 0, 클래스 1)
# =================================================================================

model = Sequential([

    Input(shape=(x_train.shape[1],)),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        64,
        activation='relu'
    ),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        2,
        activation='softmax'
    )

])

model.summary()


# =================================================================================
# 6. 1단계 : MSE 선학습
#
# 원-핫 레이블 [1,0] / [0,1] 에 대한 MSE 선학습
# =================================================================================

print()
print("=" * 80)
print("1단계 : MSE Pre-training")
print("=" * 80)


model.compile(

    loss='mse',

    optimizer=Adam(
        learning_rate=0.0001
    ),

    metrics=[
        'accuracy'
    ]

)


es_mse = EarlyStopping(

    monitor='val_loss',

    patience=50,

    mode='min',

    restore_best_weights=True,

    verbose=1

)


start_time = time.time()


history_mse = model.fit(

    x_train,
    y_train,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=300,

    batch_size=1024,

    callbacks=[
        es_mse
    ],

    verbose=1

)


# =================================================================================
# 7. MSE 학습 직후 성능
# =================================================================================

mse_pred = model.predict(
    x_test,
    batch_size=4096
)

y_test_class = np.argmax(y_test, axis=1)


mse_auc = roc_auc_score(
    y_test_class,
    mse_pred[:, 1]
)


mse_class = np.argmax(
    mse_pred,
    axis=1
)


mse_acc = accuracy_score(
    y_test_class,
    mse_class
)


print()
print("=" * 80)
print("MSE 학습 후")
print("=" * 80)

print("Accuracy :", mse_acc)
print("ROC-AUC  :", mse_auc)


# =================================================================================
# 8. 2단계 : CCE (Categorical Cross Entropy) Fine-tuning
#
# ★ 모델 새로 생성하지 않음
# ★ 기존 MSE 학습 Weight 그대로 사용
# ★ categorical_crossentropy 사용
# =================================================================================

print()
print("=" * 80)
print("2단계 : Categorical Cross Entropy Fine-tuning")
print("=" * 80)


model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(
        learning_rate=0.000001
    ),

    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]

)


es_cce = EarlyStopping(

    monitor='val_auc',

    patience=100,

    mode='max',

    restore_best_weights=True,

    verbose=1

)


history_cce = model.fit(

    x_train,
    y_train,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=1000,

    batch_size=1024,

    callbacks=[
        es_cce
    ],

    verbose=1

)


end_time = time.time()


# =================================================================================
# 9. 최종 평가
# =================================================================================

print()
print("=" * 80)
print("최종 평가")
print("=" * 80)


result = model.evaluate(
    x_test,
    y_test,
    batch_size=4096,
    verbose=1
)


print()
print("Keras evaluate")
print("loss     :", result[0])
print("accuracy :", result[1])
print("auc      :", result[2])


# =================================================================================
# 10. 예측
# =================================================================================

y_pred_probability = model.predict(
    x_test,
    batch_size=4096
)


print()
print("예측 확률 (Softmax 출력 클래스 1 확률 일부)")
print(y_pred_probability[:20, 1])


# =================================================================================
# 11. Accuracy
# =================================================================================

y_pred_class = np.argmax(
    y_pred_probability,
    axis=1
)


acc_score = accuracy_score(
    y_test_class,
    y_pred_class
)


# =================================================================================
# 12. ROC-AUC
# =================================================================================

auc_score = roc_auc_score(
    y_test_class,
    y_pred_probability[:, 1]
)


print()
print("=" * 80)
print("최종 결과")
print("=" * 80)

print("MSE 단계 AUC :", mse_auc)
print("최종 Accuracy:", acc_score)
print("최종 ROC-AUC :", auc_score)

print(
    "걸린 시간 :",
    round(end_time - start_time, 2),
    "초"
)


# =================================================================================
# 13. Kaggle 제출
# =================================================================================

y_submit = model.predict(
    test_scaled,
    batch_size=4096
)[:, 1]


submission['target'] = y_submit


submission.to_csv(
    path + 'submission_softmax_cce.csv'
)


print()
print("submission_softmax_cce.csv 생성 완료")

print(submission.head(20))