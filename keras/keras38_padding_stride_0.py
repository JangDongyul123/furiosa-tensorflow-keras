import numpy as np
import pandas as pd
from tensorflow.keras.datasets import MNIST
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델 구성
model = Sequential()
model.add(Conv2D( filters=10, kernel_size=(2,2), input_shape=(10,10,1), padding= 'same', strides=2
))
model.add(MaxPooling2D())
model.add(Conv2D(filters= 9, kernel_size= (3,3), padding= 'valid', strides=2
))

model.summary()