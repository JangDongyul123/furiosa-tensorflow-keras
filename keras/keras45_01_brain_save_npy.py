import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 훈련 속도 향상을 위한 데이터 추출 및 NPY 저장
start_time = time.time()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=120,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
)

x_train, y_train = train_data[0][0], train_data[0][1]
x_test, y_test = test_data[0][0], test_data[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(f"데이터 변환 시간 : {round(time.time() - start_time, 2)} 초")

# NPY 파일로 저장 (로드 속도 개선)
np_path = './_data/brain_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras45_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras45_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras45_01_y_test.npy', arr=y_test)

print("NPY 저장 완료.")
