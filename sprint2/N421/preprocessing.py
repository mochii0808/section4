import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")

tokenizer = Tokenizer(nlp.vocab)


# 예시문
sent1 = "I am a student."
sent2 = "J is the alphabet that follows i."
sent3 = "Is she a student trying to become a data scientist?"

sent_list = [sent1, sent2, sent3]
  
#------------------------------------------------------------------------------------

# 정규표현식

import re


# 전처리 함수 제작
def lower_and_regex(sentence):
    # 정규식 적용
    tokens = re.sub(r'[a-zA-Z0-9]', '', sentence)

    # 소문자로 변경
    tokens = tokens.lower().split()
    return tokens

# 전처리된 문장 저장
prep_sent_list = [lower_and_regex(x) for x in sent_list]

# 모든 토큰 담을 리스트
total_tokens_prep = []


# 문장 리스트에서 모든 단어를 토큰으로 만들어 저장
for i, prep_sent in enumerate(tokenizer.pipe(prep_sent_lst)):
                             #tokenizer.pipe : 문장 리스트 담아 사용
                             #생성자
    sent_token_prep = [token.text for token in prep_sent]
    total_tokens_prep.extend(sent_token_prep)
                     #extend : 각 항목들 삽입


# 중복 제거한 전체 토큰
# set : 중복 제거
token_set_prep = set(total_tokens_prep)


