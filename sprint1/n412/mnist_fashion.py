import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# mnist 의류 이미지 데이터 셋
from tensorflow.keras.datasets import fashion_mnist


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 데이터 정규화(이미지 데이터)
X_train, X_test = X_train / 255.0, X_test / 255.0

#-------------------------------------------------------------------------------------


# 모델링
model = Sequential([
    Flatten(input_shape=(28, 28)), # 이미지 입력 평탄화
    Dense(10, activation='softmax') # 10개 클래스
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              # categorical : 원핫 인코딩
              # sparse_categorical : 정수형 인코딩
              metrics=['accuracy'])


# 모델 구조 파악
model.summary()

