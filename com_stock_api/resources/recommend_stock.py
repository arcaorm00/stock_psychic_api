from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.member import MemberDto

import pandas as pd
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse

import numpy as np
import math





# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================



class RecommendStockPreprocessing():

    def hook_process(self, members):

        isAdmin = members['email'] == 'admin@stockpsychic.com'
        members = members[~isAdmin]
        
        # 컬럼 삭제
        members = self.drop_feature(members, 'password')
        members = self.drop_feature(members, 'name')
        members = self.drop_feature(members, 'profile')
        members = self.drop_feature(members, 'role')
        members = self.drop_feature(members, 'probability_churn')
        
        # 데이터 정제
        members = self.geography_nominal(members)
        members = self.gender_nominal(members)
        members = self.age_ordinal(members)
        members = self.drop_feature(members, 'age')
        members = self.creditScore_ordinal(members)
        members = self.balance_ordinal(members)
        members = self.estimatedSalary_ordinal(members)

        return members

    # ---------------------- 데이터 정제 ----------------------
    @staticmethod
    def drop_feature(members, feature):
        members = members.drop([feature], axis=1)
        return members

    @staticmethod
    def creditScore_ordinal(members):
        members['credit_score'] = pd.qcut(members['credit_score'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members

    @staticmethod
    def geography_nominal(members):
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        members['geography'] = members['geography'].map(geography_mapping)
        return members

    @staticmethod
    def gender_nominal(members):
        gender_mapping = {'Male': 0, 'Female': 1, 'Etc.': 2}
        members['gender'] = members['gender'].map(gender_mapping)
        return members

    @staticmethod
    def age_ordinal(members):
        members['age'] = members['age'].fillna(-0.5)
        bins = [-1, 18, 25, 30, 35, 40, 45, 50, 60, np.inf] # 범위
        labels = ['Unknown', 'Youth', 'YoungAdult', 'Thirties', 'LateThirties', 'Forties', 'LateForties', 'AtferFifties', 'Senior']
        members['AgeGroup'] = pd.cut(members['age'], bins, labels=labels)
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
        members['AgeGroup'] = members['AgeGroup'].map(age_mapping)
        return members

    @staticmethod
    def balance_ordinal(members):
        members['balance'] = pd.qcut(members['balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members

    @staticmethod
    def estimatedSalary_ordinal(members):
        members['estimated_salary'] = pd.qcut(members['estimated_salary'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members





class RecommendStocks():

    def hook_process(self, email):
        similarity = self.similarity(email)
        # print(f'similarity: \n{similarity}')

    @staticmethod
    def similarity(email):
        members = pd.read_sql_table('members', engine.connect())
        preprocessing = RecommendStockPreprocessing()
        refined_members = preprocessing.hook_process(members)
        refined_members.set_index(refined_members['email'], inplace=True)
        refined_members = refined_members.drop(['email'], axis=1)
        print(f'REFINED MEMBERS: \n{refined_members}')
        
        this_member = refined_members[refined_members.index == email].T
        print(f'This_Member: \n {this_member}')
        is_this = refined_members.index == email
        else_members = refined_members[~is_this]
        print(f'ELSE MEMBERS: \n {else_members}')

        sim_dict = {}

        for mem in else_members.index:
            row_mem = else_members[else_members.index == mem]
            # print(f'NP.ARRAY THISMEM: {this_member.to_numpy()}')
            # print(f'NP.ARRAY ROWMEM: {row_mem.to_numpy()}')
            
            main_n = np.linalg.norm(this_member)
            member_n = np.linalg.norm(row_mem)
            prod = np.dot(this_member, row_mem)
            print(f'prod: {prod}')

            sim_dict[mem] = prod/(main_n*member_n)

            if mem == '15570931@gmail.com':
                break

        return sim_dict


if __name__ == '__main__':
    rs = RecommendStocks()
    rs.hook_process(email='15660679@gmail.com')
    




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

