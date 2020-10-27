from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from com_stock_api.resources.member import MemberDto

import pandas as pd
import json


import os
baseurl = os.path.abspath(os.path.dirname(__file__))
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from com_stock_api.utils.file_helper import FileReader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from flask_restful import Resource, reqparse

class MemberChurnPredDto(db.Model):
    
    __tablename__ = 'member_churn_preds'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    probability_churn: float = db.Column(db.FLOAT, nullable=False)

    def __init__(self, id, email, probability_churn):
        self.id = id
        self.email = email
        self.prob_churn = probability_churn

    def __repr__(self):
        return f'MemberChurnPred(id={self.id}, email={self.email}, prob_churn={self.prob_churn})'

    @property
    def json(self):
        return {'id': self.id, 'email': self.email, 'prob_churn': self.probability_churn}

class MemberChurnPredVo:
    id: int = 0
    email: str = ''
    probability_churn: float = 0.0







class MemberChurnPredDao(MemberChurnPredDto):

    @classmethod
    def count(cls):
        return cls.query.count()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, member):
        sql = cls.query.filter(cls.email.like(member.email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(member):
        db.session.add(member)
        db.session.commit()
    
    @staticmethod
    def insert_many():
        print('*******insert many')
        service = MemberChurnPredService()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(MemberChurnPredDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @staticmethod
    def modify_member(member):
        db.session.add(member)
        db.commit()
    
    @classmethod
    def delete_member(cls, email):
        data = cls.query.get(email)
        db.session.delete(data)
        db.session.commit()

# mcp_dao = MemberChurnPredDao()
# MemberChurnPredDao.insert_many()





# =====================================================================
# =====================================================================
# ============================== service ==============================
# =====================================================================
# =====================================================================






class MemberChurnPredService:

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
        self.create_model()
        self.train_model()
        self.eval_model()
        self.debug_model()
        refined_data = self.get_prob()
        refined_data = refined_data.rename({'Email': 'email', 'Prob_churn': 'probability_churn'}, axis='columns')

        return refined_data


    def create_train(self, this):
        return this.drop('Exited', axis=1)

    def create_label(self, this):
        return this['Exited']

    def get_data(self):
        self.reader.context = os.path.join(baseurl, 'saved_data')
        self.reader.fname = 'member_refined.csv'
        data = self.reader.csv_to_dframe()
        emails = data['Email']
        self.email_list = emails.tolist()
        data = data.drop(['Email'], axis=1)
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

    # ---------- 확률 ----------
    member_id_list = []
    email_list = []
    model_y_list = []
    true_y_list = []
    prob_churn_list = []

    def get_prob(self):
        self.reader.context = os.path.join(baseurl, 'saved_data')
        self.reader.fname = 'member_refined.csv'
        data = self.reader.csv_to_dframe()
        data = data.drop(['Email'], axis=1)
        print(f'***************{data}')
        y = data['Exited']
        member_ids = data['CustomerId']
        data = self.create_train(data)
        
        data = data.to_numpy()

        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        new_model = LogisticRegression()
        new_model.fit(self.x_train, self.y_train)

        refine_data = scaler.transform(data)
        model_answers = new_model.predict(refine_data)
        
        self.member_id_list = member_ids.tolist()
        self.model_y_list = model_answers.tolist()
        # print(self.model_y_list)
        self.true_y_list = y.tolist()

        proba = new_model.predict_proba(refine_data)
        print(proba)
        print(proba[1][0])
        churn_proba = np.array([proba[i][1] for i in range(len(proba))])
        # print(churn_proba)

        self.prob_churn_list = churn_proba.tolist()

        self.save_proba_file(data, churn_proba, proba)
        refined_data = self.save_proba_database()
        return refined_data

    def save_proba_file(self, data, churn_proba, proba):
        columns = ['회원ID', '모델 답', '실제 답', '이탈 가능성']
        refined_dict = {
            'MemberID': self.member_id_list,
            'Email': self.email_list,
            'Model_Y': self.model_y_list,
            'True_Y': self.true_y_list,
            'Prob_churn': self.prob_churn_list
        }

        refined_data = pd.DataFrame(refined_dict)
        print(refined_data)
        
        context = os.path.join(baseurl, 'saved_data')
        refined_data.to_csv(os.path.join(context, 'member_churn_prob.csv'), index=False)
        print('file saved')

    def save_proba_database(self):
        refined_dict = {
            'Email': self.email_list,
            'Prob_churn': self.prob_churn_list
        }

        refined_data = pd.DataFrame(refined_dict)

        return refined_data





# =====================================================================
# =====================================================================
# ============================ controller =============================
# =====================================================================
# =====================================================================







class MemberChurnPred(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('prob_churn', type=float, required=True, help='This field cannot be left blank')

    def post(self):
        data = self.parser.parse_args()
        pred = MemberChurnPredDto(data['id'], data['email'], data['prob_churn'])
        try:
            pred.save()
        except:
            return {'message': 'An error occured inserting the MemberCurnPreds'}, 500
        
        return pred.json(), 201

    def get(self, id):
        pred = MemberChurnPredDao.find_by_id(id)
        if pred:
            return pred.json()

        return {'message': 'MemberChurnPrediction not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        pred = MemberChurnPredDao.find_by_id(id)

        pred.prob_churn = data['prob_churn']
        pred.save()
        return pred.json()

class MemberChurnPreds(Resource):

    def get(self):
        return {'member_churn_preds': list(map(lambda memberChurnPred: memberChurnPred.json(), MemberChurnPredDao.find_all()))}