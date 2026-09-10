import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# =================================================================================
# 1. 데이터
# =================================================================================

datasets = fetch_covtype()

x = datasets['data']       # 입력 데이터
y = datasets['target']     # 정답 데이터

print(x.shape)  # (581012, 54)
print(y.shape)  # (581012,)

print(np.unique(y, return_counts=True))

# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
#
# 클래스가 1부터 7까지 7개 존재한다. (0은 없음)


# =================================================================================
# 2. train / test 분리
# =================================================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=66,
    shuffle=True,
    stratify=y
)

print(x_train.shape, x_test.shape)  # (464809, 54), (116203, 54)
print(y_train.shape, y_test.shape)  # (464809,), (116203,)


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
fetch_covtype 데이터는 정답 클래스가 1부터 7까지이다.

TensorFlow의 to_categorical을 사용할 때 주의할 점:
to_categorical은 값이 1~7이더라도, 0번 클래스가 있는 것으로 인식해서
크기가 8인 원-핫 벡터([0, 1, 0, ..., 0])를 만들어버린다. (shape: N, 8)

반면 pandas의 get_dummies나 sklearn의 OneHotEncoder는 
존재하는 1~7 클래스에 대해서만 7개의 열을 생성한다. (shape: N, 7)

여기서는 pandas의 get_dummies를 사용하여 7개의 열로 인코딩한다.
'''

# -------------------------------------------------------------------------
# 원-핫 방법 2. pandas - get_dummies
# -------------------------------------------------------------------------

y_train = pd.get_dummies(y_train, dtype=int).values
y_test = pd.get_dummies(y_test, dtype=int).values

print(y_train.shape)  # (464809, 7)
print(y_test.shape)   # (116203, 7)


# -------------------------------------------------------------------------
# 원-핫 방법 1. TensorFlow - to_categorical
# -------------------------------------------------------------------------

# from tensorflow.keras.utils import to_categorical
# 
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
# print(y_train.shape) # (464809, 8) 이 됨 주의!


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
        500,
        activation='relu',
        input_dim=x.shape[1]
    )
)

model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))


model.add(
    Dense(
        y_train.shape[1],  # 7
        activation='softmax'
    )
)

'''
출력층의 뉴런이 7개인 이유:
get_dummies로 원-핫 인코딩한 y의 열 개수가 7개이기 때문이다.

만약 to_categorical을 사용했다면 열 개수가 8이 되므로 Dense(8)로 해야 한다.
'''


# =================================================================================
# 5. 컴파일
# =================================================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# =================================================================================
# 6. EarlyStopping
# =================================================================================

es = EarlyStopping(
    monitor='val_loss',
    patience=300,
    mode='auto',
    restore_best_weights=True
)


# =================================================================================
# 7. 훈련
# =================================================================================

start_time = time.time()

# 데이터가 58만 개로 크기 때문에 batch_size를 늘려준다.
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=10240, # 102400은 너무 커서 OOM(메모리 부족) 또는 학습 저하를 유발할 수 있으므로 적절한 크기로 조정
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
# acc : 0.93 이상


# =================================================================================
# 9. 예측
# =================================================================================

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print("예측 인덱스 (일부)")
print(y_pred[:10])


# =================================================================================
# 10. 원-핫 정답을 다시 변환
# =================================================================================

'''
pandas get_dummies로 변환한 결과를 np.argmax 하면 열 인덱스(0부터 6까지)가 나온다.
마찬가지로 모델의 예측값(y_pred)도 0부터 6까지의 인덱스를 출력한다.

실제 클래스는 1부터 7까지이지만,
인덱스(0~6)끼리 비교하여 정확도를 구하는 데는 문제가 없으므로 그대로 사용한다.
'''

y_test = np.argmax(y_test, axis=1)

print("실제 인덱스 (일부)")
print(y_test[:10])


# =================================================================================
# 11. 정확도 계산
# =================================================================================

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print("걸린 시간 :", round(end_time - start_time), "초")