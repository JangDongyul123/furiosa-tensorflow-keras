import numpy as np

x1 = np.array([1,2,3]) # (3,)
print("x1 = ", x1.shape)

x2 = np.array([[1,2,3]]) # (1,3)
print("x2 = ", x2.shape)

x3 = np.array([[1,2],[3,4]]) # (2,2)
print("x3 = ", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6]]) # (3,2)
print("x4 = ", x4.shape)

#np.array([[1,2],[3,4],[5,6,7]]) 이런 건 안됨 X

#np.array([[1,2],[3,4],[5,6,]]) 이런 건 됨 O  ,는 더 쓰고 싶은 게 있으면 써봐라는 개념이라서

x5 = np.array([[[1,2],[3,4],[5,6]]]) # (1,3,2)
print("x5 = ", x5.shape)

x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]) # (2,2,2)
print("x6 = ", x6.shape)

x7 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]]) # (1,1,1,2,5)
print("x7 = ", x7.shape)

x8 = np.array([[[1,2,3]],[[4,5,6]]])
print("x8 = ", x8.shape)

