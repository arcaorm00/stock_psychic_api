from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, func
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


# ---------------------- database 저장 파일을 모델링용 데이터로 전환
# ==> database에서 회원 정보 바로 받아와서 전처리 후 모델링까지 한번에 가는 걸로
class MemberModelingDataPreprocessing:
    
    def __init__(self):
        self.filereader = FileReader()

    def hook_process(self):
        this = self.filereader
        this.context = os.path.join(baseurl, 'saved_data')
        # 데이터 정제 전 database data
        this.fname = 'member_detail.csv'
        members = this.csv_to_dframe()
        this.train = members
        
        # 컬럼 삭제
        this.train.drop(this.train.loc[this.train['CustomerId']==0].index, inplace=True)
        print(this.train)
        this = self.drop_feature(this, 'RowNumber') # 열 번호 삭제
        this = self.drop_feature(this, 'Surname') # 이름 삭제
        # this = self.drop_feature(this, 'Email') # 이메일 삭제
        this = self.drop_feature(this, 'Role') # 권한 삭제
        this = self.drop_feature(this, 'Password') # 비밀번호 삭제
        this = self.drop_feature(this, 'Profile') # 프로필 이미지 삭제
        
        
        # 데이터 정제
        this = self.geography_nominal(this)
        this = self.gender_nominal(this)
        this = self.age_ordinal(this)
        this = self.drop_feature(this, 'Age')
        this = self.creditScore_ordinal(this)
        this = self.balance_ordinal(this)
        this = self.estimatedSalary_ordinal(this)

        # 고객의 서비스 이탈과 각 칼럼간의 상관계수
        # self.correlation_member_secession(this.train)

        # label 컬럼 재배치
        this = self.columns_relocation(this)

        # 정제 데이터 저장
        # self.save_preprocessed_data(this)
        
        # print(this)
        return this.train
        

    # 고객의 서비스 이탈과 각 칼럼간의 상관계수
    def correlation_member_secession(self, members):
        member_columns = members.columns
        member_correlation = {}
        for col in member_columns:
            cor = np.corrcoef(members[col], members['Exited'])
            # print(cor)
            member_correlation[col] = cor
        # print(member_correlation)
        '''
        r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
        r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
        r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
        r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
        r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
        r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
        r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계

        result:

        {'CustomerId': array([[ 1.        , -0.00624799], [-0.00624799,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'CreditScore': array([[ 1.        , -0.02709354], [-0.02709354,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Geography': array([[1.        , 0.15377058], [0.15377058, 1.        ]]), ==> 약한 양적 선형관계
        'Gender': array([[1.        , 0.10651249], [0.10651249, 1.        ]]), ==> 약한 양적 선형관계
        'Age': array([[1.        , 0.28532304], [0.28532304, 1.        ]]), ==> 약한 양적 선형관계
        'Tenure': array([[ 1.        , -0.01400061], [-0.01400061,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Balance': array([[1.        , 0.11853277], [0.11853277, 1.        ]]), ==> 약한 양적 선형관계
        'NumOfProducts': array([[ 1.        , -0.04781986], [-0.04781986,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'HasCrCard': array([[ 1.        , -0.00713777], [-0.00713777,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'IsActiveMember': array([[ 1.        , -0.15612828], [-0.15612828,  1.        ]]), ==> 약한 음적 선형관계
        'EstimatedSalary': array([[1.        , 0.01300995], [0.01300995, 1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'Exited': array([[1., 1.], [1., 1.]]), 
        'AgeGroup': array([[1.        , 0.21620629], [0.21620629, 1.        ]])} ==> 약한 양적 선형관계
        '''


    # ---------------------- 데이터 정제 ----------------------
    @staticmethod
    def create_train(this):
        return this.train.drop('Exited', axis=1)

    @staticmethod
    def create_label(this):
        return this.train['Exited']

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def surname_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        this.train['CreditScore'] = pd.qcut(this.train['CreditScore'], 11, labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def geography_nominal(this):
        # print(this.train['Geography'].unique()) 
        # ==> ['France' 'Spain' 'Germany']
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        this.train['Geography'] = this.train['Geography'].map(geography_mapping)
        return this

    @staticmethod
    def gender_nominal(this):
        gender_mapping = {'Male': 0, 'Female': 1, 'Etc': 2}
        this.train['Gender'] = this.train['Gender'].map(gender_mapping)
        this.train = this.train
        return this

    @staticmethod
    def age_ordinal(this):
        train = this.train
        train['Age'] = train['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 범위
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'YoungAdult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Baby', 
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'YoungAdult',
            6: 'Adult',
            7: 'Senior'
        }

        # for x in range(len(train['AgeGroup'])):
        #     if train['AgeGroup'][x] == 'Unknown':
        #         train['AgeGroup'][x] = age_title_mapping[train[''][x]]
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1, 
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'YoungAdult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        this.train = train
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
        this.train['Balance'] = pd.qcut(this.train['Balance'].rank(method='first'), 11, labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def numOfProducts_ordinal(this):
        return this

    @staticmethod
    def hasCrCard_numeric(this):
        return this

    @staticmethod
    def isActiveMember_numeric(this):
        return this

    @staticmethod
    def estimatedSalary_ordinal(this):
        this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 11, labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this


    # ---------------------- 파일 저장 ---------------------- 
    # def save_preprocessed_data(self, this):
    #     this.context = os.path.join(basedir, 'saved_data')
    #     this.train.to_csv(os.path.join(this.context, 'member_refined.csv'))
    #     print('file saved')

    # ---------------------- label 컬럼 위치 조정 ---------------------- 
    def columns_relocation(self, this):
        cols = this.train.columns.tolist()
        # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'Exited', 'AgeGroup']
        # cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
        cols =  (cols[:-3] + cols[-1:]) + cols[-3:-1]
        cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
        # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'AgeGroup', 'Exited']
        this.train = this.train[cols]
        print(this.train)
        return this






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





Session = openSession()
session = Session()
member_churn_preprocess = MemberModelingDataPreprocessing()




class MemberChurnPredDao(MemberChurnPredDto):

    @staticmethod
    def bulk():
        Session = openSession()
        session = Session()
        member_churn_preprocess = MemberModelingDataPreprocessing()
        df = member_churn_preprocess.hook_process()
        print(df.head())
        session.bulk_insert_mappings(MemberChurnPredDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def count():
        return session.query(func.count(MemberChurnPredDto.email)).one()
    
    @staticmethod
    def save(member_pred):
        new_pred = MemberChurnPredDto()
        db.session.add(new_pred)
        db.session.commit()


# mcp_dao = MemberChurnPredDao()
# MemberChurnPredDao.insert_many()

# MemberChurnPredDao.bulk()





# =====================================================================
# =====================================================================
# ============================== modeling =============================
# =====================================================================
# =====================================================================




class MemberChurnPredModel(object):

    x_train: object = None
    y_train: object = None
    x_validation: object = None
    y_validation: object = None
    x_test: object = None
    y_test: object = None
    model: object = None

    def __init__(self):
        self.reader = FileReader()
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'member_churn')

    def hook(self):
        self.get_data()
        self.create_model()
        self.train_model()
        self.eval_model()
        self.debug_model()
        refined_data = self.get_prob()
        refined_data = refined_data.rename({'Email': 'email', 'Prob_churn': 'probability_churn'}, axis='columns')

        
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

        saver = tf.train.Saver()
        saver.save(self.model, self.path+'/member_churn.ckpt')
        print('모델 저장 완료')
    
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
        # refined_data.to_csv(os.path.join(context, 'member_churn_prob.csv'), index=False)
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
# ============================== service ==============================
# =====================================================================
# =====================================================================





class MemberChurnPredService:

    ...





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
        member_churn_preds = MemberChurnPredDao.find_all()
        return member_churn_preds