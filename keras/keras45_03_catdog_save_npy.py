import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Cat & Dog 데이터 추출 및 NPY 저장
# 메모리 절약 및 I/O 병목 현상 방지를 위해 npy로 먼저 저장합니다.
start_time = time.time()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = './_data/image/cat-dog/training_set/'
path_test = './_data/image/cat-dog/test_set/'

# 대용량 이미지 로드
train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=8005,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=2023,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

x_train, y_train = train_data[0][0], train_data[0][1]
x_test, y_test = test_data[0][0], test_data[0][1]

print(f"데이터 변환 시간 : {round(time.time() - start_time, 2)} 초")

# NPY 저장 (.gitignore 필수)
np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras45_03_x_train.npy', arr=x_train)
np.save(np_path + 'keras45_03_y_train.npy', arr=y_train)
np.save(np_path + 'keras45_03_x_test.npy', arr=x_test)
np.save(np_path + 'keras45_03_y_test.npy', arr=y_test)

print("Cat/Dog NPY 파일 저장 완료.")
