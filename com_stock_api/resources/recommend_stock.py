from com_stock_api.ext.db import db, openSession
from com_stock_api.resources.member import MemberDto

import pandas as pd
import json

from typing import List
from flask_restful import Resource, reqparse

class RecommendStockDto(db.Model):

    __tablename__ = 'recommend_stocks'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    stock_type: str = db.Column(db.String(50), nullable=True)
    stock_id: int = db.Column(db.Integer, nullable=False)

    def __init__(self, email, stock_type, stock_id):
        self.email = email
        self.stock_type = stock_type
        self.stock_id = stock_id

    def __repr__(self):
        return f'id={self.id}, email={self.email}, stock_type={self.stock_type}, stock_id={self.stock_id}'

    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_id': self.stock_id
        }

class RecommendStockVo:
    id: int = 0
    email: str = ''
    stock_type: str =''
    stock_id: int = 0







class RecommendStockDao(RecommendStockDto):

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, recommend):
        sql = cls.query.filter(cls.id==recommend.id)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, recommend):
        sql = cls.query.filter(cls.email == recommend.email)
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
    def post():
        data = parser.parse_args()
        recommend = RecommendStockDto(data['id'], data['email'], data['stock_type'], data['stock_id'])
        try:
            recommend.save()
        except:
            return {'message': 'An error occured inserting the RecommendStocks'}, 500
        return recommend.json(), 201
    
    def get(self, id):
        try:
            recommend = RecommendStockDao.find_by_id(id)
            if recommend:
                return recommend.json()
        except Exception as e:
            return {'message': 'RecommendStocks not found'}, 404

    @staticmethod
    def put(id):
        args = parser.parse_args()
        print(f'RecommendStock {args} updated')
        try:
            RecommendStockDao.update(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'RecommendStock not found'}, 404

    @staticmethod
    def delete(id):
        try:
            RecommendStockDao.delete(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

class RecommendStocks(Resource):

    def post(self):
        rs_dao = RecommendStockDao()
        rs_dao.insert_many('recommend_stocks')

    def get(self):
        data = RecommendStockDao.find_all()
        return data, 200