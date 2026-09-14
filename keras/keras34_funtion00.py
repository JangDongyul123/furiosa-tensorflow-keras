from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 순차적 모델

# model = Sequential()
# model.add(Dense(10, input_shape = (3,)))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.summary()

###################################################

# 2-2. 함수형 모델

input1 = Input(shape = (3,))
dense1 = Dense(10, name = 'ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10, name = 'ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)
model2 = Model(inputs = input1, outputs = output1)
model2.summary()

###################################################
# 같은 조건이면 순차적 모델이랑 함수형 모델이랑 결과가 똑같다.