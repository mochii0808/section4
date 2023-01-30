from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 레이블 사용 X
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.

#-------------------------------------------------------------------------------------------


# AutoEncoder 모델링


# 잠재 벡터 크기
LATENT_DIM = 64


# 오토인코더 클래스
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   

        # Encoder 부분
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        # Decoder 부분
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = Autoencoder(LATENT_DIM)
model.compile(optimizer='adam', loss='mse')

# 인코딩, 디코딩된 이미지
encoded_imgs = model.encoder(x_test).numpy()
decoded_imgs = model.decoder(encoded_imgs).numpy()

#-------------------------------------------------------------------------------------------


# DAE (Denoise Auto Encoder)


# random noise 추가
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

# cliping
# tensor 숫자들 중 min~max 초과하는 숫자를 min, max로 변경
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


# DAE 모델링
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)), 
            layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = Denoise()
model.compile(optimizer='adam', loss='mse')

#-------------------------------------------------------------------------------------------

