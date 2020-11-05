from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, and_
import pandas as pd
import json

import os
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader
import numpy as np

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse

from com_stock_api.ext.db import db, openSession, engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, func

import pandas as pd
import json

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



'''
 * @ Module Name : member.py
 * @ Description : Member & Member Churn Prediction
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     sqlalchemy, flask_restful
 * @ 수정일         수정자                      수정내용
 *   ------------------------------------------------------------------------
 *   2020.11.01     곽아름      새 멤버 insert시 이탈 확률 값 할당 로직 추가
 *   2020.11.01     곽아름      멤버 데이터셋 insert시 이탈 확률 값 할당 로직 추가
 *   2020.11.02     곽아름      이탈 확률이 0.6 이상인 멤버 추출 메소드 추가
''' 



# member
# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================




'''
RowNumber,        레코드 행 번호
CustomerId,       고객 ID
Surname,          고객 last name
CreditScore,      신용점수
Geography,        지역
Gender,           성별
Age,              나이
Tenure,           존속 기간
Balance,          잔액
NumOfProducts,    구매 물품 수
HasCrCard,        신용카드 여부
IsActiveMember,   활성 고객 여부
EstimatedSalary,  급여 수준
Exited            서비스 탈퇴 여부
'''

# ---------------------- raw data를 database 저장용 데이터로 전환
class MemberDBDataProcessing:

    def __init__(self):
        # print(f'basedir: {basedir}')
        self.fileReader = FileReader()
        self.datapath = os.path.abspath(os.path.dirname(__file__))

    def process(self, data):
        service = self
        this = self.fileReader
        this.context = self.datapath
        this.train = service.new_model(data)
        this = service.age_ordinal(this)
        this = service.estimatedSalary_ordinal(this)
        this = service.password_nominal(this)
        this = service.email_nominal(this)
        this = service.role_nominal(this)
        this = service.set_profileimage(this)
        this = service.set_probability(this)

        # 데이터베이스 속성명과 컬럼명 일치 작업
        this = service.drop_feature(this, 'RowNumber')
        this = service.drop_feature(this, 'CustomerId')
        this.train = this.train.rename({'Surname': 'name', 'CreditScore': 'credit_score', 'Geography': 'geography', 
        'Gender': 'gender', 'Age': 'age', 'Tenure': 'tenure', 'Balance': 'balance', 'NumOfProducts': 'stock_qty', 'HasCrCard': 'has_credit', 'IsActiveMember': 'is_active_member', 
        'EstimatedSalary': 'estimated_salary', 'Password': 'password', 'Email': 'email', 'Role': 'role', 'Profile': 'profile', 'Probability_churn': 'probability_churn', 'Exited': 'exited'}, axis='columns')

        return this.train

    def new_model(self, payload) -> object:
        this = self.fileReader
        this.context = os.path.join(self.datapath, 'data')
        this.fname = payload
        # print(f'*****{this.context + this.fname}')
        return pd.read_csv(os.path.join(this.context, this.fname))

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def surname_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        return this

    @staticmethod
    def geography_nominal(this):
        return this

    @staticmethod
    def gender_nominal(this):
        return this

    @staticmethod
    def age_ordinal(this):
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
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
        return this

    # 비밀번호 추가 (1234 통일)
    @staticmethod
    def password_nominal(this):
        this.train['Password'] = '1234'
        return this

    # 이메일 추가 (CustomerId@gmail.com 통일)
    @staticmethod
    def email_nominal(this):
        this.train['Email'] = ''
        for idx in range(len(this.train)):
            if this.train.loc[idx, 'CustomerId'] == 0:
                this.train.loc[idx, 'Email'] = 'admin@stockpsychic.com'
            else:
                this.train.loc[idx,'Email'] = str(this.train.loc[idx,'CustomerId']) + '@gmail.com'
        return this
    
    # 권한 추가 (모두 회원 권한)
    @staticmethod
    def role_nominal(this):
        this.train['Role'] = ''
        for idx in range(len(this.train)):
            if this.train.loc[idx, 'CustomerId'] == 0:
                this.train.loc[idx, 'Role'] = 'ROLE_ADMIN'
            else:
                this.train.loc[idx, 'Role'] = 'ROLE_USER'
        return this

    # 프로필 이미지 추가
    @staticmethod
    def set_profileimage(this):
        this.train['Profile'] = 'noimage.png'
        return this

    # 이탈 확률 추가
    @staticmethod
    def set_probability(this):
        this.train['Probability_churn'] = -1
        return this




# prediction
# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================





class MemberModelingDataPreprocessing:
    
    def __init__(self):
        self.filereader = FileReader()
        self.isNewMember = False

    def hook_process(self, member_data):
        this = self.filereader
        
        members = member_data
        this.train = members

        if isinstance(this.train, MemberDto):
            m = this.train
            _data = {'email': m.email, 'password': m.password, 'name': m.name, 'geography': m.geography, 'gender': m.gender, 'age': int(m.age), 'profile': m.profile, 
            'tenure': int(m.tenure), 'stock_qty': int(m.stock_qty), 'balance': float(m.balance), 'has_credit': int(m.has_credit), 'credit_score': int(m.credit_score), 'is_active_member': int(m.is_active_member),
                'estimated_salary': float(m.estimated_salary), 'role': m.role, 'probability_churn': float(m.probability_churn), 'exited': int(m.exited)}
            this.train = pd.DataFrame([_data])
            self.isNewMember = True 
            members_data = pd.read_sql_table('members', engine.connect())
            this.train = pd.concat([members_data, this.train], ignore_index=True)

        # isAdmin = this.train['email'] == 'admin@stockpsychic.com'
        # this.train = this.train[~isAdmin]
        
        # 컬럼 삭제
        this = self.drop_feature(this, 'email')
        this = self.drop_feature(this, 'password')
        this = self.drop_feature(this, 'name')
        this = self.drop_feature(this, 'profile')
        this = self.drop_feature(this, 'role')
        this = self.drop_feature(this, 'probability_churn')
        
        # 데이터 정제
        this = self.geography_nominal(this)
        this = self.gender_nominal(this)
        this = self.age_ordinal(this)
        this = self.drop_feature(this, 'age')
        this = self.creditScore_ordinal(this)
        this = self.balance_ordinal(this)
        this = self.estimatedSalary_ordinal(this)

        # 고객의 서비스 이탈과 각 칼럼간의 상관계수
        # self.correlation_member_secession(this.train)

        # label 컬럼 재배치
        this = self.columns_relocation(this)

        if self.isNewMember:
            this.train = this.train.tail(1)
            this.train.index = [0]
            print(f'EVERYTHING IS DONE: \n{this.train}')

        return this.train
        

    # 고객의 서비스 이탈과 각 칼럼간의 상관계수
    def correlation_member_secession(self, members):
        member_columns = members.columns
        member_correlation = {}
        for col in member_columns:
            cor = np.corrcoef(members[col], members['exited'])
            member_correlation[col] = cor
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
    def drop_feature(this, feature):
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def name_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        this.train['credit_score'] = pd.qcut(this.train['credit_score'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def geography_nominal(this):
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        this.train['geography'] = this.train['geography'].map(geography_mapping)
        return this

    @staticmethod
    def gender_nominal(this):
        gender_mapping = {'Male': 0, 'Female': 1, 'Etc.': 2}
        this.train['gender'] = this.train['gender'].map(gender_mapping)
        this.train = this.train
        return this

    @staticmethod
    def age_ordinal(this):
        train = this.train
        train['age'] = train['age'].fillna(-0.5)
        bins = [-1, 18, 25, 30, 35, 40, 45, 50, 60, np.inf] # 범위
        labels = ['Unknown', 'Youth', 'YoungAdult', 'Thirties', 'LateThirties', 'Forties', 'LateForties', 'AtferFifties', 'Senior']
        train['AgeGroup'] = pd.cut(train['age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Youth', 
            2: 'YoungAdult',
            3: 'Thirties',
            4: 'LateThirties',
            5: 'Forties',
            6: 'LateForties',
            7: 'AtferFifties',
            8: 'Senior'
        }
        age_mapping = {
            'Unknown': 0,
            'Youth': 1, 
            'YoungAdult': 2,
            'Thirties': 3,
            'LateThirties': 4,
            'Forties': 5,
            'LateForties': 6,
            'AtferFifties': 7,
            'Senior': 8
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

        this.train = train
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
        this.train['balance'] = pd.qcut(this.train['balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
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
        this.train['estimated_salary'] = pd.qcut(this.train['estimated_salary'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    # ---------------------- label 컬럼 위치 조정 ---------------------- 
    def columns_relocation(self, this):
        cols = this.train.columns.tolist()
        cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
        this.train = this.train[cols]
        return this


# ---------------------- MemberPro ---------------------- 
class MemberPro:
    def __init__(self):
        self.db_data_process = MemberDBDataProcessing()
        
    def hook(self):
        ddp = self.db_data_process
        database_df = ddp.process('member_dataset.csv')
        return database_df






# member
# =====================================================================
# =====================================================================
# =====================      modeling      ============================
# =====================================================================
# =====================================================================





class MemberDto(db.Model):

    __tablename__ = "members"
    __table_args__ = {"mysql_collate": "utf8_general_ci"}

    email: str = db.Column(db.String(100), primary_key=True, index=True)
    password: str = db.Column(db.String(50), nullable=False)
    name: str = db.Column(db.String(50), nullable=False)
    profile: str = db.Column(db.String(200), default='noimage.png')
    geography: str = db.Column(db.String(50))
    gender: str = db.Column(db.String(10))
    age: int = db.Column(db.Integer)
    tenure: int = db.Column(db.Integer, default=0)
    stock_qty: int = db.Column(db.Integer, default=0)
    balance: float = db.Column(db.FLOAT, default=0.0)
    has_credit: int = db.Column(db.Integer)
    credit_score: int = db.Column(db.Integer)
    is_active_member: int = db.Column(db.Integer, nullable=False, default=1)
    estimated_salary: float = db.Column(db.FLOAT)
    role: str = db.Column(db.String(30), nullable=False, default='ROLE_USER')
    probability_churn: float = db.Column(db.FLOAT, default=-1)
    exited: int = db.Column(db.Integer, nullable=False, default=0)

    tradings = db.relationship('TradingDto', back_populates='member', lazy='dynamic')
    boards = db.relationship('BoardDto', back_populates='member', lazy='dynamic')
    comments = db.relationship('CommentDto', back_populates='member', lazy='dynamic')

    def __init__(self, email, password, name, profile, geography, gender, age, tenure, stock_qty, balance, has_credit, credit_score, is_active_member, estimated_salary, role, probability_churn, exited):
        self.email = email
        self.password = password
        self.name = name
        self.profile = profile
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.stock_qty = stock_qty
        self.balance = balance
        self.has_credit = has_credit
        self.credit_score = credit_score
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary
        self.role = role
        self.probability_churn = probability_churn
        self.exited = exited

    def __repr__(self):
        return 'Member(email={}, password={},'\
        'name={}, profile={}, geography={}, gender={}, age={}, tenure={}, stock_qty={}, balance={},'\
        'hasCrCard={}, credit_score={}, isActiveMember={}, estimatedSalary={}, role={}, probability_churn={}, exited={}'\
        .format(self.email, self.password, self.name, self.profile, self.geography, self.gender, self.age, self.tenure, self.stock_qty, self.balance, self.has_credit, self.credit_score, self.is_active_member, self.estimated_salary, self.role, self.probability_churn, self.exited)

    @property
    def json(self):
        return {
            'email': self.email,
            'password': self.password,
            'name': self.name,
            'profile': self.profile,
            'geography': self.geography,
            'gender': self.gender,
            'age': self.age,
            'tenure': self.tenure,
            'stock_qty': self.stock_qty,
            'balance': self.balance,
            'has_credit': self.has_credit,
            'credit_score': self.credit_score,
            'is_active_member': self.is_active_member,
            'estimated_salary': self.estimated_salary,
            'role': self.role,
            'probability_churn': self.probability_churn,
            'exited': self.exited
        }

class MemberVo:
    email: str = ''
    password: str = ''
    name: str = ''
    profile: str = ''
    geography: str = ''
    gender: str = ''
    age: int = 0
    tenure: int = 0
    stock_qty: int = 0
    balance: float = 0.0
    has_credit: int = 0
    credit_score: int = 0
    is_active_member: int = 1
    estimated_salary: float = 0.0
    role: str = 'ROLE_USER'
    probability_churn: float = 0.0
    exited: int = 0








Session = openSession()
session = Session()

class MemberDao(MemberDto):

    def __init__(self):
        ...
    
    @classmethod
    def count(cls):
        return session.query(func.count(MemberDto.email)).one()

    @classmethod
    def find_all(cls):
        sql = cls.query.filter(cls.exited != 1)
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, email):
        sql = cls.query.filter(cls.email.like(email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_high_proba_churn(cls):
        sql = cls.query.filter(and_(cls.exited != 1, cls.probability_churn > 0.6)).order_by(cls.probability_churn.desc())
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_name(cls, member):
        sql = cls.query.filter(cls.name.like(member.name))
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def login(cls, member):
        sql = cls.query.filter(cls.email.like(member.email))\
            .filter(cls.password.like(member.password))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print('=======================================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(member):
        db.session.add(member)
        db.session.commit()

    @staticmethod
    def insert_many():
        service = MemberPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())

        # 저장된 모델로 멤버 이탈 확률 구하기
        mmdp = MemberModelingDataPreprocessing()
        refined_members = mmdp.hook_process(df)
        print(f'REFINED_MEMBERS: \n{refined_members}')
        refined_members = refined_members.drop('exited', axis=1)
        refined_members = [np.array(refined_members, dtype = np.float32)]
        print(f'REFINED_MEMBERS AFTER NUMPY ARRAY: \n{refined_members}')

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'member')
        new_model = tf.keras.models.load_model(os.path.join(path, 'member_churn.h5'))

        model_pred = new_model.predict(refined_members)
        print(f'MODEL PREDICTION: {model_pred}')

        df['probability_churn'] = model_pred
        print(f'LAST DATAFRAME: {df}')

        session.bulk_insert_mappings(MemberDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
        
    
    @staticmethod
    def update(member):
        Session = openSession()
        session = Session()
        member = session.query(MemberDto)\
        .filter(MemberDto.email==member.email)\
        .update({MemberDto.password: member['password'], MemberDto.name: member['name'], MemberDto.profile: member['profile'], MemberDto.geography: member['geography'],
        MemberDto.gender: member['gender'], MemberDto.age: member['age'], MemberDto.tenure: member['tenure'], MemberDto.stock_qty: member['stock_qty'], MemberDto.balance: member['balance'],
        MemberDto.has_credit: member['has_credit'], MemberDto.credit_score: member['credit_score'], MemberDto.is_active_member: member['is_active_member'], MemberDto.estimated_salary: member['estimated_salary'],
        MemberDto.role: member['role'], MemberDto.probability_churn: member['probability_churn'], MemberDto.exited: member['exited']})
        session.commit()
        session.close()
    
    @classmethod
    def delete(cls, email):
        data = cls.query.get(email)
        db.session.delete(data)
        db.session.commit()
        db.session.close()






# prediction
# =====================================================================
# =====================================================================
# =====================      training      ============================
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
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'member')

    def hook(self):
        self.get_data()
        self.create_model()
        self.train_model()
        self.eval_model()
        self.save_model()
        self.debug_model()

        
    def create_train(self, this):
        return this.drop('Exited', axis=1)

    def create_label(self, this):
        return this['Exited']

    def get_data(self):
        data = pd.read_sql_table('members', engine.connect())

        # 전처리
        modeling_data_process = MemberModelingDataPreprocessing()
        refined_data = modeling_data_process.hook_process(data)
        data = refined_data.to_numpy()

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
        checkpoint_path = os.path.join(self.path, 'member_churn_train', 'cp.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        
        self.model.fit(self.x_train, self.y_train, epochs=50, callbacks=[cp_callback], validation_data=(self.x_validation, self.y_validation), verbose=1)  

        self.model.load_weights(checkpoint_path)
        
        self.model.save_weights(checkpoint_path.format(epoch=0))
        
    # 모델 평가
    def eval_model(self):
        print('********** eval model **********')

        loss, acc = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=2)
        print('Accuracy of Model: {:5.2f}%'.format(100 * acc))

    def save_model(self):
        self.model.save(os.path.join(self.path, 'member_churn.h5'))
 
    # 모델 디버깅
    def debug_model(self):
        print(f'self.train_data: \n{(self.x_train, self.y_train)}')
        print(f'self.validation_data: \n{(self.x_validation, self.y_validation)}')
        print(f'self.test_data: \n{(self.x_test, self.y_test)}')






# member
# =====================================================================
# =====================================================================
# =====================      service       ============================
# =====================================================================
# =====================================================================




# member에 post가 있을 때마다 수행해서 나온 값을 해당 멤버 proba_churn 컬럼에 넣어줘야 함
class MemberChurnPredService(object):

    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'member')
    
    geography: int = 0
    gender: int = 0
    tenure: int = 0
    stock_qty: int = 0
    balance: float = 0.0
    has_credit: int = 0
    credit_score: int = 0
    is_active_member: int = 0
    estimated_salary: float = 0.0
    AgeGroup: int = 0
    probability_churn: float = 0

    def assign(self, member):

        mmdp = MemberModelingDataPreprocessing()
        refined_member = mmdp.hook_process(member)
    
        self.geography = refined_member['geography'][0]
        self.gender = refined_member['gender'][0]
        self.tenure = refined_member['tenure'][0]
        self.stock_qty = refined_member['stock_qty'][0]
        self.balance = refined_member['balance'][0]
        self.has_credit = refined_member['has_credit'][0]
        self.credit_score = refined_member['credit_score'][0]
        self.is_active_member = refined_member['is_active_member'][0]
        self.estimated_salary = refined_member['estimated_salary'][0]
        self.AgeGroup = refined_member['AgeGroup'][0]

    def predict(self):
        new_model = tf.keras.models.load_model(os.path.join(self.path, 'member_churn.h5'))
        new_model.summary()

        data = [[self.geography, self.gender, self.tenure, self.stock_qty, self.balance, self.has_credit,
         self.credit_score, self.is_active_member, self.estimated_salary, self.AgeGroup], ]
        print(f'predict data: \n {data}')
        data = np.array(data, dtype = np.float32)
        
        pred = new_model.predict(data)

        return pred

if __name__ == "__main__":
    mcp = MemberChurnPredModel()
    mcp.hook()





# member
# =====================================================================
# =====================================================================
# =====================      resource      ============================
# =====================================================================
# =====================================================================




parser = reqparse.RequestParser()
parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('password', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('name', type=str, required=False)
parser.add_argument('profile', type=str, required=False, default='noimage.png')
parser.add_argument('geography', type=str, required=False)
parser.add_argument('gender', type=str, required=False)
parser.add_argument('age', type=int, required=False)
parser.add_argument('tenure', type=int, required=False)
parser.add_argument('stock_qty', type=int, required=False)
parser.add_argument('balance', type=float, required=False)
parser.add_argument('has_credit', type=int, required=False)
parser.add_argument('credit_score', type=int, required=False)
parser.add_argument('is_active_member', type=int, required=False)
parser.add_argument('estimated_salary', type=float, required=False)
parser.add_argument('role', type=str, required=False)
parser.add_argument('probability_churn', type=float, required=False)
parser.add_argument('exited', type=int, required=False)

class Member(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Member{args["email"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len(params) == 0:
            return 'No parameter'
        
        params_str = ''
        for key in params.key():
            params_str += 'key: {}, value: {}\n' .format(key, params[key])
        print(f'params_str: {params_str}')
        return {'code': 0, 'message': 'SUCCESS'}, 200    

    @staticmethod
    def get(email: str):
        try:
            member = MemberDao.find_by_email(email)
            print(f'member: {member}')
            if member:
                return member
        except Exception as e:
            print(e)
            return {'message': 'Member not found'}, 404
    
    @staticmethod
    def put(email: str):
        args = parser.parse_args()
        print(f'Member {args} updated')
        try:
            MemberDao.update(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Member not found'}, 404
    
    @staticmethod
    def delete(email: str):
        print('member delete')
        try:
            MemberDao.delete(email)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Member not found'}, 404

class Members(Resource):

    def post(self):
        m_dao = MemberDao()
        m_dao.insert_many('members')

    def get(self):
        data = MemberDao.find_all()
        return data, 200
    
class Auth(Resource):

    def post(self):
        body = request.get_json()
        print(f'body: {body}')
        member = MemberDto(**body)

        if len(MemberDao.find_by_email(member.email)) > 0:
            return {'message': 'already exist'}, 500

        mcp = MemberChurnPredService()
        mcp.assign(member)
        prediction = mcp.predict()
        
        prediction = round(prediction[0, 0], 5)
        print(f'PREDICTION: {prediction}')
        member.probability_churn = float(prediction)

        MemberDao.save(member)

        email = member.email
        return {'email': str(email)}, 200
    
class Access(Resource):

    def post(self):
        print('=============== member_api.py / Access')
        args = parser.parse_args()
        print(f'args: {args}')
        member = MemberVo()
        member.email = args.email
        member.password = args.password
        print(f'email: {member.email}')
        print(f'password: {member.password}')
        data = MemberDao.login(member)
        if data[0]['exited'] == 0:
            return data[0], 200
        else:
            print('탈퇴한 계정입니다!')
            return {'message': 'Member not found'}, 500

class HighChurnMembers(Resource):

    def get(self):
        members = MemberDao.find_high_proba_churn()
        return members, 200