# gensim 패키지

import gensim
import gensim.downloader as api


# 구글 뉴스 말뭉치 학습 벡터
wv = api.load('word2vec-google-news-300')

#------------------------------------------------------------------------------------------------------


# 단어 확인

for idx, word in enumerate(wv.index_to_key):
    if idx == 10:
        break

    print(f"word #{idx}/{len(wv.index_to_key)} is '{word}'")
         #단어의 인덱스 / 전체 단어 개수         is  '단어


#-------------------------------------------------------------------------------------------------------


# 단어 간 유사도 파악
# .similarity

pairs = [
    ('car', 'minivan'),   
    ('car', 'bicycle'),  
    ('car', 'airplane'),
    ('car', 'cereal'),    
    ('car', 'democracy')
]

for w1, w1, in pairs:
    print(f'{w1} ======= {w2}\t  {wv.similarity(w1, w2):.2f}')

#==============================================================

# 유사한 단어 추출
# .most_similar

# 'car' + 'minivan'
for i, (word, similarity) in enumerate(wv.most_similar(positive=['car', 'minivan'], topn=5)):
    print(f"Top {i+1} : {word}, {similarity}")

# 'king' + 'women' - 'men'
print(wv.most_similar(positive=['king', 'women'], negative=['men'], topn=1))

# 'walking' + 'swam' - 'walked'
print(wv.most_similar(positive=['walking', 'swam'], negative=['walked'], topn=1))

#================================================================

# 무관한 단어 추출
# .doesnt_match

print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))

#-------------------------------------------------------------------------------------------------------


# 문장 분류 수행

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

#=================================================================

# 인덱스 -> 텍스트 변경 함수

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
     
#=================================================================

# tokenizer 학습

sentences = [decode_review(idx) for idx in X_train]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)


# 단어 집합 크기

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

#================================================================

# 패딩 처리

X_encoded = tokenizer.texts_to_sequences(sentences)
                    # texts_to_sequences : 텍스트를 시퀀스로(key 리스트)

max_len = max(len(sent) for sent in X_encoded)

print(f'학습 데이터에 있는 문서의 평균 토큰 수: {np.mean([len(sent) for sent in X_train], dtype=int)}')

# max_len = 평균보다 긴 400으로 설정
maxlen_pad = 400

X_train = pad_sequences(X_encoded, maxlen=maxlen_pad, padding='post')
        # pad_sequences : 최대 길이만큼 문장 길이 채우기
y_train = np.array(y_train)


# word2vec 가중치 행렬
embedding_matrix = np.zeros((vocab_size, 300))

# 단어의 임베딩 벡터 반환 함수
def get_vector(word):
    if word in wv: # 단어가 vocab에 있다면
        return wv[word]
    else:
        return None
    
for word, i in tokenizer.word_index.items():
    temp = get_vector(word)
    if temp is not None:
        embedding_matrix[i] = temp

#-------------------------------------------------------------------------------------------------------


# 훈련

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten


# 모델
model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen_pad, trainable=False))
model.add(GlobalAveragePooling1D()) 
        # 입력되는 단어 벡터의 평균 계산
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)


# X_test 패딩
test_sentences = [decode_review(idx) for idx in X_test]
X_test_encoded = tokenizer.texts_to_sequences(test_sentences)
X_test=pad_sequences(X_test_encoded, maxlen=400, padding='post')


y_test=np.array(y_test)
     

model.evaluate(X_test, y_test)