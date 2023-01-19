# 코사인 유사도

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity('''행렬''')

#------------------------------------------------------------------------------------


# K-최근접 이웃

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
                     # 이웃한 5개 단어

# tfidf 문서-단어 행렬 이용                     
nn.fit(dtm_tfidf) 

# 2번째 문서와 가장 가까운 5가지 문서(본인 포함)의 거리, 인덱스 출력
nn.kneighbors([dtm_tfidf.iloc[2]])

