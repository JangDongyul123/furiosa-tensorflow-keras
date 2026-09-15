# 36 - 2번 카피

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
# (60000, 28, 28), (60000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (60000, 28, 28, 1)

print(x_test.shape, y_test.shape) 
# (10000, 28, 28), (10000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (10000, 28, 28, 1)

# print(x_train[0])

print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

#### 스케일링 1 - min - max 스케일링
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

#### 스케일링 2 - max ABS 스케일링
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

