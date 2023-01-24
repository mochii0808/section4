import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


# 데이터셋
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

#---------------------------------------------------------------------------------------------------------------

# Process
# 영어 <-> 스페인어 번역 모델

'''
전처리
   ↓
토큰화
   ↓
데이터 분할
   ↓
파라미터 설정
   ↓
인코더 구현
   ↓
Attention 구현
   ↓
디코더 구현
   ↓
손실함수 설정
   ↓
결과 및 시각화
'''


#---------------------------------------------------------------------------------------------------------------


# 전처리


# 유니코드 파일을 아스키코드로 변환
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


# 문자열 전처리 함수
def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # 글자와 문장부호 사이 공간 생성
  # eg: "he is a boy." => "he is a boy ."
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # (a-z, A-Z, ".", "?", "!", ",")를 제외하고 모두 제거
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # 문장에 시작 토큰과 종료 토큰 삽입
  w = ' ' + w + ' '
  return w


# 전처리 적용 후 (영어, 스페인어)쌍 반환 함수
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w) for w in line.split('\t')]
                for line in lines[:num_examples]]

  return zip(*word_pairs)

#---------------------------------------------------------------------------------------------------------------


# 토큰화
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer


# 2개 언어이기 때문에 tokenizer가 input/targ 언어별로 따로 필요
def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

#---------------------------------------------------------------------------------------------------------------


# 데이터 분할


# load할 dataset의 개수를 30000으로 설정
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)


# target과 input의 최대 길이 구하기(문장 최대 길이)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]


# 8:2 비율로 train-test split 진행
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

#---------------------------------------------------------------------------------------------------------------


# 파라미터 설정

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

# tf.data.Dataset 
# 배치 구성, 데이터셋 셔플, 윈도우 구현, 변환 함수 적용 등 다양한 기능을 제공
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#---------------------------------------------------------------------------------------------------------------


# 인코더 구현

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    # |x| = (batch_sz, seq_len)
    x = self.embedding(x) # |x| = (batch_sz, seq_len, embedding_dim)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
  
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
  
#---------------------------------------------------------------------------------------------------------------


# Attention 구현

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    # W1(query_with_time_axis) == (batch_size, 1, units)
    # W2(values) == (batch_size, max_len, units)
    # W1(query_with_time_axis) + W2(values) == (batch_size, max_len, units)
    # V(tf.nn.tanh(W1(query_with_time_axis) + W2(values))) == (batch_size, max_len, 1)

    attention_weights = tf.nn.softmax(score, axis=1) # attention_weights == (batch_size, max_length, 1)

    context_vector = attention_weights * values # context_vector == (batch_size, max_len, hidden_size)
    context_vector = tf.reduce_sum(context_vector, axis=1) # context_vector == (batch_size, hidden_size)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)

#---------------------------------------------------------------------------------------------------------------

# 디코더 구현

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # x == (batch_size, 1)
    # hidden == (batch_size, hidden_size)
    # enc_output shape == (batch_size, max_length, hidden_size)

    context_vector, attention_weights = self.attention(hidden, enc_output)
    # context_vector == (batch_size, hidden_size)
    # attention_weights == (batch_size, max_length, 1)

    x = self.embedding(x) # x == (batch_size, 1, embedding_dim)

    # tf.expand_dims(context_vector, 1) == (batch_size, 1, hidden_size)
    # tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) == (batch_size, 1, embedding_dim+hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    # output == (batch_size, 1, hidden_size)
    # state == (batch_size, hidden_size)
    output, state = self.gru(x)

    
    output = tf.reshape(output, (-1, output.shape[2])) # output  == (batch_size * 1, hidden_size)

    x = self.fc(output) # x == (batch_size, vocab)

    return x, state, attention_weights
  
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

#---------------------------------------------------------------------------------------------------------------


# 손실 함수

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

#---------------------------------------------------------------------------------------------------------------


# 결과 및 시각화


def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


# Attention 가중치 시각화 함수
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
     

# 문장 번역 함수
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))

  # 번역된 문장 출력
  print('Predicted translation: {}'.format(result)) 

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

  # 단어의 관계 히트맵 출력
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))
