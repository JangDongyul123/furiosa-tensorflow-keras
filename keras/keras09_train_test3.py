#09_1 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 훈련 데이터와 평가 데이터를 앞 부분과 뒷 부분으로 슬라이싱으로 나눈 방법은 잘못되었다.
# 왜냐하면 데이터의 범위는 전체범위이기 때문에 앞부분에 대한 훈련 만으로 뒷 부분에 대한 평가를 하는 것도 옳지 않다.
# (다른 범위로 훈련하고 또 다른 범위로 테스트했으니. w 기울기 오차가 많이 생기는 건 당연하다. 또 평가를 신뢰할 수 없다.)
# 훈련도 전체 범위로, 테스트도 전체범위로 즉 둘 다 같은 범위 내에서 랜덤으로 나눠야한다.
# 또 같은 범위로 훈련시키고, 테스트했기 때문에 평가를 신뢰할 수 있다.

# [검색] 7:3 으로 나눈다.
# 힌트: 사이킷런
x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                train_size=0.7,  
                # test_size = 0.3만 적는 것과 train_size=0.7만 적는 것과 같다. 
                # test_size = 0.3, train_size=0.7 둘 다 같이 적어도 된다.
                # test_size = 0.3, train_size=0.8 적으면 오버플로우 문제 생긴다.
                # test_size = 0.2, train_size=0.7 적는 것은 실행 가능하다.
                # train_size 파라미터의 디폴트값은 0.75 (안적으면 75%)
                # test_size 파라미터의 디폴트값은 0.25 (안적으면 25%)
                shuffle=True, # 디폴트 섞는다.
                random_state=333 # 랜덤시드라고도 부른다. 
                # 랜덤 난수표에서 이 난수에 맞게 데이터를 섞어준다. 
                # 즉, random_state가 고정이면 훈련 데이터와 테스트 데이터는 동일한 고정된 수치로 뽑아준다.
                # competition 데이터는 이상한 숫자를 줘서 random_state만 잘 설정해서 훈련해도 1등하는 경우가 있음
)
print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

# shift+delete 한 라인 통째로 삭제
# 예측값

#2. 모델구성
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)
result = model.predict(np.array([[11]]))
print('result: ', result)
