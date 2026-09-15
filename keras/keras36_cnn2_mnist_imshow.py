import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
# (60000, 28, 28), (60000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (60000, 28, 28, 1)

print(x_test.shape, y_test.shape) 
# (10000, 28, 28), (10000,)
# 실제로는 흑백데이터라서 채널(channel)이 1개 더 필요하다. -> (10000, 28, 28, 1)

# print(x_train[0])

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

print(pd.value_counts(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[50000], cmap='gray')
#gray 뺴면 칼라로 나오는데, 흑백 데이터를 plt가 자동으로 마음대로 컬러 데이터를 넣은 것이다.getattr
plt.show()

