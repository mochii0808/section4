import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")

tokenizer = Tokenizer(nlp.vocab)

# 아마존 리뷰 데이터
df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/amazon/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_sample.csv')

#------------------------------------------------------------------------------------

# 어간 추출

from nltk.stem import PorterStemmer

ps = PorterStemmer()


# 예시
words = ['wolf', 'wolves']

for word in words:
    print(ps.stem(word))

#========================================

# stemming 적용

tokens = []

for doc in df['tokens']:
    doc_tokens = []

    # 문장 토큰에 대해 어간 추출
    for token in doc:
        doc_tokens.append(ps.stem(token))

    tokens.append(doc_tokens)


#------------------------------------------------------------------------------------

# 표제어 추출
# spacy 사용(.lemma_)


# 예시
sent = "The social wolf. Wolves are complex."

doc = nlp(sent)

for token in doc:
    print(token.text, '|' , token.lemma_)
                            # spacy 내장 .lemma_ 메서드