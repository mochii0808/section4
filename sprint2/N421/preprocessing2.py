import pandas as pd
import re

# 아마존 리뷰 데이터
df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/amazon/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_sample.csv')
    
#------------------------------------------------------------------------------------


# 횟수 카운팅

from collections import Counter

word_counts = Counter()

# 아마존 리뷰 데이터
df['reviews.text'].apply(lambda x: word_counts.update(x))
                                  #.update : 토큰 등장 횟수 카운트

#------------------------------------------------------------------------------------


# Squarify

import squarify
import matplotlib.pyplot as plt

squarify.plot(sizes='''등장 비율''', label='''등장 단어''')
plt.axis('off')
plt.show()

#------------------------------------------------------------------------------------


# Spacy
# 요소 색인화하여 저장

import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")

tokenizer = Tokenizer(nlp.vocab)

#=================================


# Spacy 전처리

tokens = []

for doc in tokenizer.pipe(df['reviews.text']):
    doc_tokens = [re.sub(r"[^a-z0-9]", "", token.text.lower()) for token in doc]
    tokens.append(doc_tokens)

df['tokens'] = tokens
df['tokens'].head()

#------------------------------------------------------------------------------------


# 불용어(Stop words) 처리


# spacy 기본 제공 불용어
print(nlp.Defaults.stop_words)

#================================


# 불용어 제거

tokens = [] # 전체 토큰 담을 리스트

for doc in tokenizer.pipe(df['reviews.text']):
    
    doc_tokens = [] # 문장 토큰 담을 리스트

    for token in doc:
        if (token.is_stop == False) & (token.is_punct == False):
            # 토큰이 불용어가 아니거나 & 구두점이 아니면 

            doc_tokens.append(token.text.lower())
            # 소문자로 변환하여 저장
    
    tokens.append(doc_tokens)

#================================================================


# 불용어 커스터마이징

# .union([])
stp_wds = nlp.Defaults.stop_words.union(['batteries','I', 'amazon', 'i', 'Amazon'])


tokens = []
for doc in tokenizer.pipe(df['reviews.text']):
    doc_tokens = [] 
    for token in doc:
        if token.text.lower() not in stp_wds:
            # 토큰이 불용어 리스트에 없다면 저장

            doc_tokens.append(token.text.lower())
    tokens.append(doc_tokens)

#------------------------------------------------------------------------------------

# 트리밍

# 토큰 빈도 데이터 제작 후 희소 데이터 삭제