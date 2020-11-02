from com_stock_api.ext.db import db, openSession
from com_stock_api.resources.member import MemberDto

import pandas as pd
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse





# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================



class RecommendStockPreprocessing():

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
    def surname_nominal(this):
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

    # @staticmethod
    def age_ordinal(self, this):
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


# =====================================================================
# =====================================================================
# ===================        modeling         =========================
# =====================================================================
# =====================================================================



class RecommendStockDto(db.Model):

    __tablename__ = 'recommend_stocks'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    stock_type: str = db.Column(db.String(50), nullable=False)
    # stock_id: int = db.Column(db.Integer, nullable=False)
    stock_ticker: str = db.Column(db.String(100), nullable=False)

    def __init__(self, email, stock_type, stock_ticker):
        self.email = email
        self.stock_type = stock_type
        # self.stock_id = stock_id
        self.stock_ticker = stock_ticker

    def __repr__(self):
        return f'id={self.id}, email={self.email}, stock_type={self.stock_type}, stock_ticker={self.stock_ticker}'

    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_ticker': self.stock_ticker
        }

class RecommendStockVo:
    id: int = 0
    email: str = ''
    stock_type: str =''
    stock_ticker: str = ''







class RecommendStockDao(RecommendStockDto):

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, recommend):
        sql = cls.query.filter(cls.id == recommend.id)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, email):
        sql = cls.query.filter(cls.email == email)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))


    @staticmethod
    def save(recommend_stock):
        db.session.add(recommend_stock)
        db.session.commit()
    
    @staticmethod
    def modify_recommend_stock(recommend_stock):
        Session = openSession()
        session = Session()
        trading = session.query(RecommendStockDto)\
        .filter(RecommendStockDto.id==recommend_stock.id)\
        .update({RecommendStockDto.stock_type: recommend_stock['stock_type'], RecommendStockDto.stock_id: recommend_stock['stock_id']})
        session.commit()
        session.close()

    @classmethod
    def delete_recommend_stock(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        db.session.close()





# =====================================================================
# =====================================================================
# ============================ controller =============================
# =====================================================================
# =====================================================================





parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('stock_type', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('stock_id', type=str, required=True, help='This field cannot be left blank')

class RecommendStock(Resource):

    @staticmethod
    def post(id):
        body = request.get_json()
        print(f'body: {body}')
        recomm_stock = RecommendStockDto(**body)
        RecommendStockDao.save(recomm_stock)
        return {'recomm_stock': str(recomm_stock.id)}, 200
    
    @staticmethod
    def get(id):
        args = parser.parse_args()
        try:
            recomm_stock = RecommendStockDao.find_by_email(args['email'])
            if recomm_stock:
                return recomm_stock
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404

    @staticmethod
    def put(id):
        args = parser.parse_args()
        print(f'Recommend Stock {args} updated')
        try:
            RecommendStockDao.modify_recommend_stock(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404
   
    @staticmethod
    def delete(id):
        try:
            RecommendStockDao.delete_recommend_stock(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404
    
class Boards(Resource):
    
    def post(self):
        rs_dao = RecommendStockDao()
        rs_dao.insert_many('recommend_stocks')

    def get(self):
        data = RecommendStockDao.find_all()
        return data, 200

