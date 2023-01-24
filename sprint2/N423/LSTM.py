# 라이브러리, 데이터 

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import IPython

path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    # 말뭉치 개수
    text = f.read().lower()

# 문자 개수
chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# 파라미터
max_features = 20000
maxlen = 80
batch_size = 32

#-------------------------------------------------------------------------------------------------------------


# 모델링

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)),
                dropout = 0.2, recurrent_dropout=0.2))
                             # recurrent dropout : 현재 정보에 영향을 받는 parameter에만 dropout
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer=optimizer)

#-------------------------------------------------------------------------------------------------------------


