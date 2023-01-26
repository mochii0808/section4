import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# 컬러이미지 데이터셋
from tensorflow.keras.datasets import cifar10


(X_train, y_train), (X_test, y_test) = cifar10.load_data() # 데이터 로드
X_train, X_test= X_train.astype('float32')/255., X_test.astype('float32')/255. # 정규화
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2) # 분할

#---------------------------------------------------------------------------------------------------------------


# 모델링

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', strides=(2,2)))
    # 합성곱층(필터 개수, 필터 크기, 패딩 방식, 스트라이드 크기 ...)
model.add(MaxPooling2D(2,2))
    # 최대 풀링층
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#---------------------------------------------------------------------------------------------------------------


# 전이학습

from tensorflow.keras.applications.vgg16 import VGG16

# 사전학습 모델 호출
pretrained_model = VGG16(weights='imagenet', include_top=False)