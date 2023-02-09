# 라이브러리
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터셋
df = pd.read_csv('~')

# -----------------------------------------------------------------------------------

# 모델링 과정
# 신경망 구축 -> compile -> fit -> score


# 신경망 구축
# 1-1. Sequential 1
model = tf.keras.models.Sequential([
    # 입력층 기입안하면 자동입력
    tf.keras.layers.Dense(10, activation='relu'), # 은닉층
    tf.keras.layers.Dense(1, activation='softmax') # 출력층
])

# 1-2. Sequential 2
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='softmax'))


# 2 Functional
input = tf.keras.layers.Input(shape=(2,)) # 입력층에 크기 정의
output = tf.keras.layers.Dense(1, activation='sigmoid')(input) # 이전층은 다음층의 입력
model = tf.keras.models.Model(inputs=input, outputs=output)

#==========================================================

# compile
# 옵티마이저, 손실함수, 지표
model.compiel(optimizer='sgd',
              loss='crossentropy',
              metrics=['accuracy'])

#==========================================================

# 훈련
model.fit(X_train, y_train, epochs=30)

#==========================================================

# 평가
model.evaluate(X_test, y_test, verbose=2)