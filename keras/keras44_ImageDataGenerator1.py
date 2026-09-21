import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 데이터 파이프라인 구축 (ImageDataGenerator)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # 0~255 픽셀 값을 0~1로 스케일링
    horizontal_flip=True,     # 수평 뒤집기 
    vertical_flip=True,       # 수직 뒤집기 
    width_shift_range=0.1,    # 좌우 이동 
    height_shift_range=0.1,   # 상하 이동 
    rotation_range=5,         # 이미지 회전
    zoom_range=1.2,           # 확대/축소
    shear_range=0.7,          # 기울이기
    fill_mode='nearest',      # 빈 공간 채우는 방식
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 경로 설정
path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

# 디렉토리에서 이미지 가져오기
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',      # ad, normal 2진 분류
    color_mode='grayscale',   # 흑백 이미지
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
)

print(xy_train)
print(xy_train.next())
print(xy_train[0][0].shape)   # (10, 100, 100, 1)
print(xy_train[0][1].shape)   # (10,)

print(type(xy_train))
print(type(xy_train[0]))
print(type(xy_train[0][0]))
print(type(xy_train[0][1]))