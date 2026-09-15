from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(10, (3,3,1), input_shape = (10, 10, 1)))
model.add(Conv2D(5, (2,2)))

model.summary()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 8, 8, 10)            │             100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 7, 7, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘