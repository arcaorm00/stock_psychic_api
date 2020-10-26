from com_stock_api.ext.db import db
from com_stock_api.resources.member import MemberDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from com_stock_api.naver_finance.dto import StockDto
import datetime

import pandas as pd
import json

from typing import List
from flask_restful import Resource, reqparse

class TradingDto(db.Model):

    __tablename__ = "tradings"
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    kospi_stock_id: int = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    nasdaq_stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    stock_qty: int = db.Column(db.Integer, nullable=False)
    price: float = db.Column(db.FLOAT, nullable=False)
    trading_date: str = db.Column(db.String(1000), default=datetime.datetime.now())

    def __init__(self, id, email, kospi_stock_id, nasdaq_stock_id, stock_qty, price, trading_date):
        self.id = id
        self.email = email
        self.kospi_stock_id = kospi_stock_id
        self.nasdaq_stock_id = nasdaq_stock_id
        self.stock_qty = stock_qty
        self.price = price
        self.trading_date = trading_date

    def __repr__(self):
        return 'Trading(trading_id={}, member_id={}, kospi_stock_id={}, nasdaq_stock_id={}, stock_qty={}, price={}, trading_date={})'.format(self.id, self.member_id, self.kospi_stock_id, self.nasdaq_stock_id, self.stock_qty, self.price, self.trading_date)
    
    @property
    def json(self):
        return {
            'id': self.id,
            'member_id': self.member_id,
            'kospi_stock_id': self.kospi_stock_id,
            'nasdaq_stock_id': self.nasdaq_stock_id,
            'stock_qty': self.stock_qty,
            'price': self.price,
            'trading_date': self.trading_date
        }

class TradingVo:
    id: int = 0
    email: str = ''
    kospi_stock_id: int = 0
    nasdaq_stock_id: int = 0
    stock_qty: int = 0
    price: float = 0.0
    trading_date: str = db.Column(db.String(1000), default=datetime.datetime.now())






class TradingDao(TradingDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, trading):
        sql = cls.query.filter_by(cls.id.like(trading.id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_email(cls, trading):
        sql = cls.query.filter_by(cls.email.like(trading.email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(trading):
        db.session.add(trading)
        db.session.commit()

    @staticmethod
    def modify_trading(trading):
        db.session.add(trading)
        db.session.commit()

    @classmethod
    def delete_trading(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()





# =====================================================================
# =====================================================================
# ============================ controller =============================
# =====================================================================
# =====================================================================







class Trading(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('kospi_stock_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('nasdaq_stock_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('stock_qty', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('price', type=float, required=True, help='This field cannot be left blank')
        parser.add_argument('trading_date', type=str, required=True, help='This field cannot be left blank')
        
    def post(self):
        data = self.parser.parse_args()
        trading = TradingDto(data['id'], data['email'], data['kospi_stock_id'], data['nasdaq_stock_id'], data['stock_qty'], data['price'], data['trading_date'])
        try:
            trading.save()
        except:
            return {'message': 'An error occured inserting the tradings'}, 500
        return trading.json(), 201
    
    def get(self, id):
        trading = TradingDao.find_by_id(id)
        if trading:
            return trading.json()
        return {'message': 'Trading not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        trading = TradingDao.find_by_id(id)

        trading.stock_qty = data['stock_qty']
        trading.price = data['price']
        trading.trading_date = data['trading_date']
        trading.save()
        return trading.json()

class Tradings(Resource):
    def get(self):
        return {'tradings': list(map(lambda trading: trading.json(), TradingDao.find_all()))}

    def get_by_email(self, email):
        return {'tradings': list(map(lambda trading: trading.json(), TradingDao.find_by_email(email)))}