import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터

# 이거는 학습이 안되는 데이터 X가 2덩이인데, Y도 2덩이어야 하는데 5덩이임
#x = np.array([[1,2,3,4,5],[6,7,8,9,10]])
#y = np.array([[1,2,3,4,5]])

# 그래서 이래야함 5덩이 5덩이
x = np.array([[1,2,3,4,5],[6,7,8,9,10]])
#x=x.T
x=x.transpose()
#x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape) #(5,2)
print(y.shape) #(5,)

# 2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=2)) #x는 feature가 2개이므로 inputDim은 2 행 무시 열 우선 
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1)) #y는 1,2,3 각각 숫자 1개

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=3) #배치사이즈가 크다고 좋은 게 아니고, 작다고 좋은 게 아니다.

#배치가 3이니까
#위에서부터 3덩이 먼저 잘라서
# 1번째 epochs 시작
# 1번째 훈련
# x = [1,6],[2,7],[3,8]
# y = [1,2,3]
# 2번째 훈련
# x = [4,9],[5,10]
# y = [4,5]
# 1번째 epochs 끝
# 2번째부터도 계속 동일 위에서부터 자르고 훈련

# 4. 평가 예측
loss = model.evaluate(x,y)
print('loss: ', loss)
results = model.predict(np.array([[6,11]]));
print('results: ', results)