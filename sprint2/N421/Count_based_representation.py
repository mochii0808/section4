import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

#------------------------------------------------------------------------------------


# BoW
# 빈도수(정수)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer('''stop_words, max_features 설정''')


# 예문

text = """In information retrieval, tf-idf or TFIDF, short for term frequency-inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.
The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word,
which helps to adjust for the fact that some words appear more frequently in general.
tf-idf is one of the most popular term-weighting schemes today.
A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use tf-idf."""
     

# 토큰화 
doc = nlp(text)

# 문장 리스트
sentences_list = text.split('\n')

# 어휘 사전 제작
vect.fit(sentences_list)

# 문서-토큰 행렬로 변환
dtm_count = vect.transform(sentences_list)

# 맵핑 인덱스 정보 확인
vect.vocabulary_

# numpy 행렬로 변환
dtm_count.todense()

# dataframe으로 변환
pd.DataFrame(dtm_count.todense(), columns=vect.get_feture_names()) 
                                             # get_feature_names :  어휘 사전에 등록된 단어들

#------------------------------------------------------------------------------------


# TF-IDF
# 희귀도(실수)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer('''stop_words, max_features 설정''')


# 문서-토큰 행렬
dtm_tfidf = tfidf.fit_transform(sentences_list)

# dataframe
pd.DataFrame(dtm_tfidf.todense(), columns=tfidf.get_feture_names()) 

#------------------------------------------------------------------------------------


# 파라미터 튜닝


# 토큰화 방법 설정
def tokenize(document):
    doc = nlp(document)
    return [token.lemma_.strip() for token in doc if (token.is_stop != True) and (token.is_punct != True) and (token.is_alpha == True)]

# 파라미터 설정
tfidf_tuned = TfidfVectorizer(stop_words='english',
                              tokenizer = tokenize,
                              ngram_range = (1,2),
                              # (min_n ~ max_m)개를 갖는 n-gram(n개의 연속적인 토큰)을 토큰으로 사용 

                              max_df=.7,
                              # xx% 이상 문서에 나타나는 토큰 제거

                              min_df=3
                              # 최소 n개의 문서에 나타나는 토큰 사용
                              )

# 문서-단어행렬 -> 데이터프레임
dtm_tfidf_tuned = tfidf_tuned.fit_transform(sentences_list)
dtm_tfidf_tuned = pd.DataFrame(dtm_tfidf_tuned.todense(), columns=tfidf_tuned.get_feature_names())
dtm_tfidf_tuned.head()