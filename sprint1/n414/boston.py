from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense


import numpy as np
import pandas as pd
import tensorflow as tf


# 보스턴 집값 데이터
from tensorflow.keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


# 모델링
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#-------------------------------------------------------------------------------------


# 교차 검증


# kf(KFold) 개수 지정
kf = KFold(n_splits = 5)
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


# 전체 프로세스
for train_index, val_index in kf.split(X_train, y_train):

    # 데이터 나누기
    training_data = X_train.iloc[train_index]
    validation_data = X_train.iloc[val_index]
    training_label = y_train.iloc[train_index]
    validation_label = y_train.iloc[val_index]

    # 컴파일
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    # 훈련
    model.fit(training_data, training_label, # 훈련 세트 지정
            epochs=10, batch_size=64,
            validation_data=(validation_data, validation_label) # 검증 세트 지정
    )

    # 결과 
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(results)


#------------------------------------------------------------------------------------


# 하이퍼 파라미터 튜닝 1
# GridSearch 이용


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


# 모델 제작(wrapping하기 위해 함수형태로)
def create_model(nodes=8):
                # 탐색할 하이퍼 파리미터의 초기값 지정 필수
    model = Sequential()
    model.add(Dense(nodes, input_dim=8, activation='relu'))
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# wrapping
model = KerasClassifier(model=create_model, batch_size=8, verbose=False)

# 하이퍼 파라미터 범위 지정
nodes = [16, 32, 64]
batch_size = [16, 32, 64]

# 파라미터 범위 딕셔너리 형태로
param_grid = dict(model__nodes=nodes, batch_size=batch_size)
                # 모델 내부 인자 : 더블 언더바


# Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)


# 최적의 결과 출력
print(grid_result.best_score_, grid_result.best_params_)

# 각 결과 출력
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"Means: {mean}, Stdev: {stdev} with: {param}") 

#------------------------------------------------------------------------------------

# 하이퍼 파라미터 튜닝 2
# Keras Tuner 라이브러리 이용


from tensorflow import keras

# !pip install -U keras-tuner 설치 후
import keras_tuner as kt


# 모델 제작(함수형)
def model_builder(hp):

    # keras.Sequential
    model = keras.Sequential()

    # 노드 수 
    # hp.int : 정수형 하이퍼 파라미터 범위
    hp_units = hp.int('units', min_value=32, max_value=512, step=32)
                              # 노드 수 32~512까지 32개씩 증가시키며 탐색
    model.add(Dense(units=hp_units, activation='relu'))
                    # 노드 수 범위 적용
    model.add(Dense(10, activation='softmax'))

    # 학습률
    # hp.choice : 정해진 하이퍼 파라미터 숫자
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                                                    # 학습률 범위 적용
                    loss = keras.losses.SparseCategoricalCrossentropy(), 
                    metrics = ['accuracy'])
    
    return model


# 튜너 지정
# Hyperband : 리소스 자동조절, 조기종료 사용
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt'
)

# Callback 함수 지정
# 학습이 끝날 때마다 이전 출력이 지워지도록
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


# 탐색 수행
tuner.search(X_train, y_train, 
             epochs = 10, 
             validation_data = (X_test, y_test), 
             callbacks = [ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# 최적의 하이퍼 파라미터
best_hps.get('units')
best_hps.get('learning_rate')


# 최적의 하이퍼파라미터로 재학습
model = tuner.hypermodel.build(best_hps)
model.summary

model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))