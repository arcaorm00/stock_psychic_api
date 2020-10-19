import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from com_stock_api.utils.file_helper import FileReader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class MemberChurnPred:

    x_train: object = None
    y_train: object = None
    x_validation: object = None
    y_validation: object = None
    x_test: object = None
    y_test: object = None
    model: object = None

    def __init__(self):
        self.reader = FileReader()

    def hook(self):
        self.get_data()
        # self.create_model()
        # self.train_model()
        # self.eval_model()
        # self.debug_model()
        self.get_prob()

    def create_train(self, this):
        return this.drop('Exited', axis=1)

    def create_label(self, this):
        return this['Exited']

    def get_data(self):
        self.reader.context = os.path.join(baseurl, os.path.join('member', 'saved_data'))
        self.reader.fname = 'member_refined.csv'
        data = self.reader.csv_to_dframe()
        data = data.to_numpy()
        # print(data[:60])

        table_col = data.shape[1]
        y_col = 1
        x_col = table_col - y_col
        x = data[:, 0:x_col]
        y = data[:, x_col:]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.4)

        self.x_train = x_train; self.x_validation = x_validation; self.x_test = x_test
        self.y_train = y_train; self.y_validation = y_validation; self.y_test = y_test

    
    # 모델 생성
    def create_model(self):
        print('********** create model **********')
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # output
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
 
    # 모델 훈련
    def train_model(self):
        print('********** train model **********')
        self.model.fit(x=self.x_train, y=self.y_train, 
        validation_data=(self.x_validation, self.y_validation), epochs=20, verbose=1)
    
    # 모델 평가
    def eval_model(self):
        print('********** eval model **********')
        results = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print('%s: %.3f' % (name, value))
 
    # 모델 디버깅
    def debug_model(self):
        print(f'self.train_data: \n{(self.x_train, self.y_train)}')
        print(f'self.validation_data: \n{(self.x_validation, self.y_validation)}')
        print(f'self.test_data: \n{(self.x_test, self.y_test)}')


    def get_prob(self):
        self.reader.context = os.path.join(baseurl, os.path.join('member', 'saved_data'))
        self.reader.fname = 'member_refined.csv'
        data = self.reader.csv_to_dframe()
        y = data['Exited']
        data = self.create_train(data)
        
        data = data.to_numpy()

        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        self.model = LogisticRegression()
        self.model.fit(self.x_train, self.y_train)

        refine_data = scaler.transform(data)
        model_answers = self.model.predict(refine_data)
        print(model_answers)
        print(y)
        proba = self.model.predict_proba(refine_data)
        print(proba)
        # print(proba[1][0])
        churn_proba = np.array(proba[i][1] for i in range(len(proba)))
        print(type(churn_proba))
        self.modify_feature()
        self.save_proba_file(data, churn_proba, proba)

    def modify_feature(self):
        ...

    def save_proba_file(self, data, churn_proba, proba):
        columns = ['회원ID', '모델 답', '실제 답', '이탈 가능성']
        

        


if __name__ == '__main__':
    training = MemberChurnPred()
    training.hook()