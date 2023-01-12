from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers

import os
import numpy as np
import tensorflow as tf
import keras

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 데이터 정규화
X_train = X_train / 255.
X_test = X_test / 255.

#---------------------------------------------------------------------------------

# 모델링

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(64,
          # Weight decay
          kernel_regularizer=regularizers.l2(0.01),
          activity_regularizer=regularizers.l1(0.01)),
    # Dropout
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.89),
                                                                    # 학습률 감소 파라미터
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#----------------------------------------------------------------------------------

# 조기 종료

# 파라미터 저장 경로
# epoch, 검증오차
checkpoint_filepath = 'FMbest.hdf5'

# 조기 종료 옵션
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', # 기준 지표, 접두어:'val_'
    min_delta=0, 
    patience=10, # 오차 증가 10회 이상이면 중단
    verbose=1 # 상황 출력(0: 출력x, 1: 출력o)
    )

# 파라미터 저장 옵션
save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, # 저장 경로
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, # 최적의 모델 1개만 저장
    save_weights_only=True, # 모델의 가중치만 저장
    mode='auto', # save_best_only=True인 경우 덮어씌우는 방식
                 # 'val_loss' -> 'min'
                 # 'val_accuracy' -> 'max'
    save_freq='epoch', # 저장 주기(매 epoch 마다)
    options=None
    )

model.fit(
    X_train, y_train, batch_size=32, epochs=30, verbose=1, 
    validation_data=(X_test,y_test), 
    # 저장 파라미터 호출
    callbacks=[early_stop, save_best]
    )

#----------------------------------------------------------------------------------

# 저장 모델 평가

# 저장된 모델의 가중치 호출
model.load_weights(checkpoint_filepath)

model.predict(X_test)