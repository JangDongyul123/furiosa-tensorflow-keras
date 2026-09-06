# 이 링크에서 .csv파일을 다운로드 받으셔야 합니다.
# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)  #  [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv)  #  [6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

print(submission)  #  [6493 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)



print(train_csv.info())
print(test_csv.info())
print(submission.info())

print(train_csv.describe())
############################### 결측치 확인 ###############################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())


############################### x,y 분리 ###############################
x = train_csv.drop(['casual','registered','count'], axis = 1)
print(x)
y = train_csv['count']
print(y, y.shape)


### 대회 코드이기에 여기 코드부터는 개인의 코드를 작성해야 합니다. ###