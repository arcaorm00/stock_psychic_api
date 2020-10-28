from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import json

import os
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader
import numpy as np

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse



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
    # exited: int = db.Column(db.Integer, nullable=False, default=0)

    tradings = db.relationship('TradingDto', back_populates='member', lazy='dynamic')

    def __init__(self, email, password, name, profile, geography, gender, age, tenure, stock_qty, balance, has_credit, credit_score, is_active_member, estimated_salary, role):
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
        # self.exited = exited

    def __repr__(self):
        return 'Member(member_id={}, email={}, password={},'\
        'name={}, profile={}, geography={}, gender={}, age={}, tenure={}, stock_qty={}, balance={},'\
        'hasCrCard={}, credit_score={}, isActiveMember={}, estimatedSalary={}, role={}'\
        .format(self.id, self.email, self.password, self.name, self.profile, self.geography, self.gender, self.age, self.tenure, self.stock_qty, self.balance, self.has_credit, self.credit_score, self.is_active_member, self.estimated_salary, self.role)

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
            # 'exited': self.exited
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



class MemberDao(MemberDto):

    def __init__(self):
        ...
    
    @classmethod
    def count(cls):
        return cls.query.count()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, email):
        sql = cls.query.filter(cls.email.like(email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_name(cls, member):
        sql = cls.query.filter(cls.name.like(member.name))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
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
        session.bulk_insert_mappings(MemberDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @staticmethod
    def update(member):
        print('UserDao UPDATE COPY THAT!')
        db.session.add(member)
        db.session.commit()
    
    @classmethod
    def delete(cls, email):
        data = cls.query.get(email)
        db.session.delete(data)
        db.session.commit()

# m_dao = MemberDao()
# MemberDao.insert_many()




# =====================================================================
# =====================================================================
# ============================== service ==============================
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
        # print(f'feature 드롭 전 변수: \n{this.train.columns}')
        # this = service.drop_feature(this, 'Exited')
        this = service.age_ordinal(this)
        # print(f'나이 정제 후: \n{this.train.head()}')
        this = service.estimatedSalary_ordinal(this)
        # print(f'수입 정제 후: \n{this.train.head()}')
        this = service.password_nominal(this)
        # print(f'비밀번호 추가 후: \n{this.train["Password"]}')
        this = service.email_nominal(this)
        # print(f'이메일 추가 후: \n{this.train["Email"]}')
        this = service.role_nominal(this)
        # print(f'권한 추가 후: \n{this.train["Role"]}')
        this = service.set_profileimage(this)
        self.datapath = os.path.join(self.datapath, 'saved_data')
        this.train.to_csv(os.path.join(self.datapath, 'member_detail.csv'), index=False)

        # 데이터베이스 속성명과 컬럼명 일치 작업
        this = service.drop_feature(this, 'RowNumber')
        this = service.drop_feature(this, 'CustomerId')
        this = service.drop_feature(this, 'AgeGroup')
        this = service.drop_feature(this, 'Exited')
        this.train = this.train.rename({'Surname': 'name', 'CreditScore': 'credit_score', 'Geography': 'geography', 
        'Gender': 'gender', 'Age': 'age', 'Tenure': 'tenure', 'Balance': 'balance', 'NumOfProducts': 'stock_qty', 'HasCrCard': 'has_credit', 'IsActiveMember': 'is_active_member', 
        'EstimatedSalary': 'estimated_salary', 'Password': 'password', 'Email': 'email', 'Role': 'role', 'Profile': 'profile'}, axis='columns')

        print(this.train)
        return this.train

    def new_model(self, payload) -> object:
        this = self.fileReader
        this.context = os.path.join(self.datapath, 'data')
        this.fname = payload
        print(f'*****{this.context + this.fname}')
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
        this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    # 비밀번호 추가 (임시 1234 통일)
    @staticmethod
    def password_nominal(this):
        this.train['Password'] = '1234'
        return this

    # 이메일 추가 (임시 uuid_CustomerId@gmail.com)
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
    


# ---------------------- database 저장 파일을 모델링용 데이터로 전환
# class MemberModelingDataPreprocessing:
    
#     def __init__(self):
#         self.filereader = FileReader()

#     def hook_process(self):
#         this = self.filereader
#         this.context = os.path.join(basedir, 'saved_data')
#         # 데이터 정제 전 database data
#         this.fname = 'member_detail.csv'
#         members = this.csv_to_dframe()
#         this.train = members
        
#         # 컬럼 삭제
#         this.train.drop(this.train.loc[this.train['CustomerId']==0].index, inplace=True)
#         print(this.train)
#         this = self.drop_feature(this, 'RowNumber') # 열 번호 삭제
#         this = self.drop_feature(this, 'Surname') # 이름 삭제
#         # this = self.drop_feature(this, 'Email') # 이메일 삭제
#         this = self.drop_feature(this, 'Role') # 권한 삭제
#         this = self.drop_feature(this, 'Password') # 비밀번호 삭제
#         this = self.drop_feature(this, 'Profile') # 프로필 이미지 삭제
        
        
#         # 데이터 정제
#         this = self.geography_nominal(this)
#         this = self.gender_nominal(this)
#         this = self.age_ordinal(this)
#         this = self.drop_feature(this, 'Age')
#         this = self.creditScore_ordinal(this)
#         this = self.balance_ordinal(this)
#         this = self.estimatedSalary_ordinal(this)

#         # 고객의 서비스 이탈과 각 칼럼간의 상관계수
#         # self.correlation_member_secession(this.train)

#         # label 컬럼 재배치
#         this = self.columns_relocation(this)

#         # 정제 데이터 저장
#         self.save_preprocessed_data(this)
        
#         # print(this)
        

#     # 고객의 서비스 이탈과 각 칼럼간의 상관계수
#     def correlation_member_secession(self, members):
#         member_columns = members.columns
#         member_correlation = {}
#         for col in member_columns:
#             cor = np.corrcoef(members[col], members['Exited'])
#             # print(cor)
#             member_correlation[col] = cor
#         # print(member_correlation)
#         '''
#         r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
#         r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
#         r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
#         r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
#         r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
#         r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
#         r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계

#         result:

#         {'CustomerId': array([[ 1.        , -0.00624799], [-0.00624799,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
#         'CreditScore': array([[ 1.        , -0.02709354], [-0.02709354,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
#         'Geography': array([[1.        , 0.15377058], [0.15377058, 1.        ]]), ==> 약한 양적 선형관계
#         'Gender': array([[1.        , 0.10651249], [0.10651249, 1.        ]]), ==> 약한 양적 선형관계
#         'Age': array([[1.        , 0.28532304], [0.28532304, 1.        ]]), ==> 약한 양적 선형관계
#         'Tenure': array([[ 1.        , -0.01400061], [-0.01400061,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
#         'Balance': array([[1.        , 0.11853277], [0.11853277, 1.        ]]), ==> 약한 양적 선형관계
#         'NumOfProducts': array([[ 1.        , -0.04781986], [-0.04781986,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
#         'HasCrCard': array([[ 1.        , -0.00713777], [-0.00713777,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
#         'IsActiveMember': array([[ 1.        , -0.15612828], [-0.15612828,  1.        ]]), ==> 약한 음적 선형관계
#         'EstimatedSalary': array([[1.        , 0.01300995], [0.01300995, 1.        ]]),  ==> 거의 무시될 수 있는 선형관계
#         'Exited': array([[1., 1.], [1., 1.]]), 
#         'AgeGroup': array([[1.        , 0.21620629], [0.21620629, 1.        ]])} ==> 약한 양적 선형관계
#         '''

#     # ---------------------- 데이터 정제 ----------------------
#     @staticmethod
#     def create_train(this):
#         return this.train.drop('Exited', axis=1)

#     @staticmethod
#     def create_label(this):
#         return this.train['Exited']

#     @staticmethod
#     def drop_feature(this, feature) -> object:
#         this.train = this.train.drop([feature], axis=1)
#         return this

#     @staticmethod
#     def surname_nominal(this):
#         return this

#     @staticmethod
#     def creditScore_ordinal(this):
#         this.train['CreditScore'] = pd.qcut(this.train['CreditScore'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
#         return this

#     @staticmethod
#     def geography_nominal(this):
#         # print(this.train['Geography'].unique()) 
#         # ==> ['France' 'Spain' 'Germany']
#         geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
#         this.train['Geography'] = this.train['Geography'].map(geography_mapping)
#         return this

#     @staticmethod
#     def gender_nominal(this):
#         gender_mapping = {'Male': 0, 'Female': 1}
#         this.train['Gender'] = this.train['Gender'].map(gender_mapping)
#         this.train = this.train
#         return this

#     @staticmethod
#     def age_ordinal(this):
#         train = this.train
#         train['Age'] = train['Age'].fillna(-0.5)
#         bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 범위
#         labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'YoungAdult', 'Adult', 'Senior']
#         train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
#         age_title_mapping = {
#             0: 'Unknown',
#             1: 'Baby', 
#             2: 'Child',
#             3: 'Teenager',
#             4: 'Student',
#             5: 'YoungAdult',
#             6: 'Adult',
#             7: 'Senior'
#         }

#         # for x in range(len(train['AgeGroup'])):
#         #     if train['AgeGroup'][x] == 'Unknown':
#         #         train['AgeGroup'][x] = age_title_mapping[train[''][x]]
        
#         age_mapping = {
#             'Unknown': 0,
#             'Baby': 1, 
#             'Child': 2,
#             'Teenager': 3,
#             'Student': 4,
#             'YoungAdult': 5,
#             'Adult': 6,
#             'Senior': 7
#         }
#         train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
#         this.train = train
#         return this

#     @staticmethod
#     def tenure_ordinal(this):
#         return this

#     @staticmethod
#     def balance_ordinal(this):
#         this.train['Balance'] = pd.qcut(this.train['Balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
#         return this

#     @staticmethod
#     def numOfProducts_ordinal(this):
#         return this

#     @staticmethod
#     def hasCrCard_numeric(this):
#         return this

#     @staticmethod
#     def isActiveMember_numeric(this):
#         return this

#     @staticmethod
#     def estimatedSalary_ordinal(this):
#         this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
#         return this


#     # ---------------------- 파일 저장 ---------------------- 
#     def save_preprocessed_data(self, this):
#         this.context = os.path.join(basedir, 'saved_data')
#         this.train.to_csv(os.path.join(this.context, 'member_refined.csv'))
#         print('file saved')

#     # ---------------------- label 컬럼 위치 조정 ---------------------- 
#     def columns_relocation(self, this):
#         cols = this.train.columns.tolist()
#         # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'Exited', 'AgeGroup']
#         # cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
#         cols =  (cols[:-3] + cols[-1:]) + cols[-3:-1]
#         cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
#         # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'AgeGroup', 'Exited']
#         this.train = this.train[cols]
#         print(this.train)
#         return this


# # ---------------------- MemberPro ---------------------- 
class MemberPro:
    def __init__(self):
        self.db_data_process = MemberDBDataProcessing()
        # self.modeling_data_process = MemberModelingDataPreprocessing()

    def hook(self):
        ddp = self.db_data_process
        database_df = ddp.process('member_dataset.csv')
        # mdp = self.modeling_data_process
        # mdp.hook_process()
        return database_df




# =====================================================================
# =====================================================================
# ============================ controller =============================
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
            print('inner try')
            MemberDao.update(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Member not found'}, 404
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Member {args["email"]} deleted')
        return {'code': 0, 'message': 'SUCCESS'}, 200

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
        return data[0], 200
