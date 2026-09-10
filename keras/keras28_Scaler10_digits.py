import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_digits

# acc : 1.0


# =================================================================================
# 1. 데이터
# =================================================================================

datasets = load_digits()

x = datasets['data']       # 입력 데이터
y = datasets['target']     # 정답 데이터

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)

print(np.unique(y, return_counts=True))

# (array([0, 1, 2]), array([59, 71, 48]))
#
# 클래스 0 : 59개
# 클래스 1 : 71개
# 클래스 2 : 48개


# =================================================================================
# 2. train / test 분리
# =================================================================================

'''
현재 y는 아직 원-핫 인코딩 전이다.

stratify=y의 의미:
원래 y에 존재하는 클래스 비율을
train과 test에서도 최대한 동일하게 유지해서 나눈다.
'''

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=66,
    shuffle=True,
    stratify=y
)

print(x_train.shape, x_test.shape)  # (142, 13), (36, 13)
print(y_train.shape, y_test.shape)  # (142,), (36,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
# x_train 데이터를 보유한 데이터, 
# x_test를 미래 데이터라고 가정한다.
# 나중에 대회에서도 private 데이터가 아닌 우리가 보유한 데이터를 기준으로 스케일링할 것이 아닌가?
# 그러므로 x_train 데이터를 기준으로 스케일링한다.


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x.shape, y.shape)

print(x, y)


# =================================================================================
# 3. 원-핫 인코딩
# =================================================================================

'''
다중분류의 정답이 클래스 번호로 되어 있을 때,
원-핫 인코딩을 통해 배열 형태로 변환한다.

Wine 데이터의 클래스가 3개이므로 원-핫 인코딩 후 열이 3개 생긴다.
'''

# -------------------------------------------------------------------------
# 원-핫 방법 1. TensorFlow - to_categorical
# -------------------------------------------------------------------------

# from tensorflow.keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape)  # (142, 3)
# print(y_test.shape)   # (36, 3)


# -------------------------------------------------------------------------
# 원-핫 방법 2. pandas - get_dummies
# -------------------------------------------------------------------------

y_train = pd.get_dummies(y_train, dtype='float32').values
y_test = pd.get_dummies(y_test, dtype='float32').values


# -------------------------------------------------------------------------
# 원-핫 방법 3. sklearn - OneHotEncoder
# -------------------------------------------------------------------------

# from sklearn.preprocessing import OneHotEncoder
#
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
#
# encoder = OneHotEncoder(sparse_output=False)
#
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.transform(y_test)


# =================================================================================
# 4. 모델 구성
# =================================================================================

model = Sequential()

model.add(
    Dense(
        64,
        activation='relu',
        input_dim=x.shape[1]
    )
)

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(
    Dense(
        10,
        activation='softmax'
    )
)

'''
출력층의 뉴런이 10개인 이유:
Digits 데이터셋의 클래스가 총 10개(0~9)이기 때문이다.

softmax는 다중분류의 출력층에서 주로 사용하는 활성화 함수이다.
모든 출력값의 합이 1이 되도록 각 클래스에 속할 확률을 계산해준다.
'''


# =================================================================================
# 5. 컴파일
# =================================================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

'''
categorical_crossentropy는
원-핫 인코딩된 정답을 사용하는 다중분류에서 주로 사용한다.
'''


# =================================================================================
# 6. EarlyStopping
# =================================================================================

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='auto',
    restore_best_weights=True
)


# =================================================================================
# 7. 훈련
# =================================================================================

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)

end_time = time.time()


# =================================================================================
# 8. 평가
# =================================================================================

result = model.evaluate(x_test, y_test)

print("loss :", result[0])
print("accuracy :", result[1])
# acc = 0.95 이상이면 합격


# =================================================================================
# 9. 예측
# =================================================================================

y_pred = model.predict(x_test)

print("softmax 출력 (일부)")
print(y_pred[:5])

'''
model.predict()의 결과는 softmax 출력값이다.
각 행에서 가장 큰 값의 위치(인덱스)를 찾아서 최종 예측 클래스를 구한다.
'''

y_pred = np.argmax(y_pred, axis=1)

print("예측 클래스 (일부)")
print(y_pred[:5])


# =================================================================================
# 10. 원-핫 정답을 다시 클래스 번호로 변환
# =================================================================================

'''
accuracy_score로 실제 클래스와 예측 클래스를 비교하기 위해
원-핫 인코딩된 y_test를 다시 원래 클래스 번호 형태로 바꾼다.
'''

y_test = np.argmax(y_test, axis=1)

print("실제 클래스 (일부)")
print(y_test[:5])


# =================================================================================
# 11. 정확도 계산
# =================================================================================

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print("걸린 시간 :", round(end_time - start_time), "초")