import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# =================================================================================
# 1. 데이터
# =================================================================================

datasets = load_iris()

x = datasets['data']       # 입력 데이터
y = datasets['target']     # 정답 데이터

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

print(np.unique(y, return_counts=True))

# (array([0, 1, 2]), array([50, 50, 50]))
#
# 클래스 0 : 50개
# 클래스 1 : 50개
# 클래스 2 : 50개


# =================================================================================
# 2. train / test 분리
# =================================================================================

'''
현재 y는 아직 원-핫 인코딩 전이다.

예:
y = [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]

stratify=y의 의미:
원래 y에 존재하는 클래스 비율을
train과 test에서도 최대한 동일하게 유지해서 나눈다.

Iris는

0 : 50개
1 : 50개
2 : 50개

이므로 80:20으로 나누면 대략

train:
0 : 40개
1 : 40개
2 : 40개

test:
0 : 10개
1 : 10개
2 : 10개

가 된다.

stratify의 목적은 "클래스 비율을 유지하여 데이터를 나누는 것"이다.

원-핫 인코딩의 목적은
"정답의 표현 방식을 바꾸는 것"이다.

따라서

데이터 분할
    ↓
원-핫 인코딩

순서로 처리하면 두 작업의 역할이 명확하다.
'''

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=66,
    shuffle=True,
    stratify=y
)

print(x_train.shape, x_test.shape)  # (120, 4), (30, 4)
print(y_train.shape, y_test.shape)  # (120,), (30,)


# =================================================================================
# 3. 원-핫 인코딩
# =================================================================================

'''
딥러닝에서 클래스는 정답의 종류를 의미한다.

다중분류의 정답이

0
1
2

와 같이 클래스 번호로 되어 있을 때,

원-핫 인코딩을 하면

0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]

형태로 바뀐다.

예:

원래:
[0, 0, 1, 1, 2]        shape = (5,)

원-핫 인코딩 후:

[
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
]

shape = (5, 3)

클래스가 3개이므로 열이 3개 생긴다.

원-핫 인코딩의 단점:
클래스 개수가 매우 많으면 대부분의 값이 0인 큰 배열이 만들어진다.

예를 들어 데이터가 60,000개이고 클래스가 100개라면

(60000,)
    ↓
(60000, 100)

형태가 된다.
'''


# -------------------------------------------------------------------------
# 원-핫 방법 1. TensorFlow - to_categorical
# -------------------------------------------------------------------------

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)  # (120, 3)
print(y_test.shape)   # (30, 3)


# -------------------------------------------------------------------------
# 원-핫 방법 2. pandas - get_dummies
# -------------------------------------------------------------------------

# 위의 to_categorical 방법 대신 사용할 수 있다.
# 세 가지 방법 중 하나만 사용해야 한다.

# y_train = pd.get_dummies(y_train, dtype=int).values
# y_test = pd.get_dummies(y_test, dtype=int).values
#
# print(y_train.shape)  # (120, 3)
# print(y_test.shape)   # (30, 3)

# 주의
# y_train = pd.get_dummies(y_train) 쓰면 1 0 0 대신 True False False 나옴
# 그래서 y_train = pd.get_dummies(y_train, dtype=int).values 이렇게 써라. (.values를 붙여 numpy 배열로 추출하는 것이 안전함)

# -------------------------------------------------------------------------
# 원-핫 방법 3. sklearn - OneHotEncoder
# -------------------------------------------------------------------------

# reshape의 조건: 내용과 순서가 바뀌면 안된다.

# sklearn의 OneHotEncoder는 입력을 2차원 형태로 받는다.
#
# 원래:
# y_train.shape = (120,)
#
# reshape 후:
# y_train.shape = (120, 1)
#
# 그리고 OneHotEncoder를 적용하면
# (120, 3)이 된다.
#
# 전처리기는 일반적으로 train 데이터에 fit하고,
# test 데이터에는 train에서 학습한 변환 규칙으로 transform만 한다.

# from sklearn.preprocessing import OneHotEncoder
#
# y_train = y_train.reshape(-1, 1) # -1은 '나머지 차원에 맞춰서 자동으로 크기를 조정'하라는 의미
# y_test = y_test.reshape(-1, 1) 
#
# ohe = OneHotEncoder(sparse_output=False) 
# ohe = OneHotEncoder() # default는 sparse_output=True Sparse 형태로 나온다.
# sparse_output=False는 0,1 배열로 출력
#
# y_train = ohe.fit_transform(y_train) 
# y_test = ohe.transform(y_test)
#
# print(y_train.shape)  # (120, 3)
# print(y_test.shape)   # (30, 3)


# =================================================================================
# 4. 모델 구성
# =================================================================================

model = Sequential()

model.add(
    Dense(
        1000,
        activation='relu',
        input_dim=x.shape[1]
    )
)

model.add(Dense(1000))

model.add(Dense(
    1000,
    activation='relu'
))

model.add(
    Dense(
        3,
        activation='softmax'
    )
)


'''
출력층의 뉴런이 3개인 이유:

Iris의 클래스가

0
1
2

총 3개이기 때문이다.


softmax는 다중분류의 출력층에서 주로 사용하는 활성화 함수이다.

Dense(3)에서 계산된 값이 예를 들어

[2.0, 1.0, 0.1]

이라고 하자.

softmax를 적용하면 대략

[0.659, 0.242, 0.099]

처럼 변환된다.

이 값들은 모두 0~1 사이이며

0.659 + 0.242 + 0.099 ≈ 1

즉 모든 출력값의 합이 1이 된다.


중요:

softmax 자체가

[0.659, 0.242, 0.099]

을

[1, 0, 0]

으로 바꾸는 것은 아니다.

softmax는 여기까지만 계산한다.

최종적으로 어떤 클래스를 예측했는지 선택하려면
가장 큰 값의 인덱스를 찾는 argmax를 사용한다.

np.argmax([0.659, 0.242, 0.099])

→ 0

따라서 모델은 class 0을 예측한 것이다.


만약

softmax([2.0, 2.0, 0.1])

처럼 앞의 두 값이 같다면

대략

[0.465, 0.465, 0.070]

처럼 두 클래스의 출력값도 같아진다.

np.argmax()는 최대값이 여러 개라면
가장 먼저 등장한 인덱스를 반환한다.
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

예:

정답:
[0, 1, 0]

예측:
[0.1, 0.8, 0.1]


만약 y를 원-핫 인코딩하지 않고

0
1
2

형태 그대로 사용할 경우에는

loss='sparse_categorical_crossentropy'

를 사용할 수 있다.


metrics=['accuracy']는
loss 외에 정확도를 보조 평가 지표로 보여준다.

예:

30개 중 29개를 맞혔다면

accuracy = 29 / 30
         = 0.966666...

학습 로그에서는 보통

accuracy: 0.9667

처럼 일부 소수점 자리까지만 표시된다.

accuracy 계산 자체가 반올림되는 것은 아니고,
출력 화면에서 표시 자릿수를 줄여서 보여주는 것이다.
'''


# =================================================================================
# 6. EarlyStopping
# =================================================================================

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
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
    epochs=100,
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


# =================================================================================
# 9. 예측
# =================================================================================

y_pred = model.predict(x_test)

print("softmax 출력")
print(y_pred)


'''
model.predict()의 결과는 클래스 번호가 아니라
softmax 출력값이다.

예:

[
    [0.1, 0.8, 0.1],
    [0.7, 0.2, 0.1],
    [0.2, 0.2, 0.6]
]

각 행에서 가장 큰 값의 위치를 찾으면

첫 번째 → index 1
두 번째 → index 0
세 번째 → index 2

따라서

[1, 0, 2]

가 최종 예측 클래스가 된다.
'''

y_pred = np.argmax(y_pred, axis=1)

print("예측 클래스")
print(y_pred)


# =================================================================================
# 10. 원-핫 정답을 다시 클래스 번호로 변환
# =================================================================================

print("원-핫 y_test")
print(y_test)


'''
현재 y_test는 원-핫 인코딩되어 있다.

예:

[
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
]

accuracy_score로 실제 클래스와 예측 클래스를 비교하기 위해
argmax를 사용해서 다시

[1, 0, 2]

형태로 바꾼다.
'''

y_test = np.argmax(y_test, axis=1)

print("실제 클래스")
print(y_test)


# =================================================================================
# 11. 정확도 계산
# =================================================================================

acc_score = accuracy_score(y_test, y_pred)

'''
accuracy_score라는 함수 이름을 그대로 변수 이름으로 사용하면 안 좋다.

accuracy_score = accuracy_score(y_test, y_pred)

이렇게 하면 accuracy_score라는 함수 이름이
숫자 변수로 덮어써진다.

따라서

acc_score = accuracy_score(...)

처럼 다른 변수 이름을 사용하는 것이 좋다.
'''

print('acc_score :', acc_score)
print("걸린 시간 :", round(end_time - start_time), "초")