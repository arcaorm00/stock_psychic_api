from com_stock_api.ext.db import db, openSession
from com_stock_api.resources.member import MemberDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from com_stock_api.naver_finance.dto import StockDto
import datetime

import pandas as pd
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse



'''
 * @ Module Name : trading.py
 * @ Description : Trading stock
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     flask_restful
''' 




# =====================================================================
# =====================================================================
# ===================        modeling         =========================
# =====================================================================
# =====================================================================




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

    member = db.relationship('MemberDto', back_populates='tradings')
    yhfinence = db.relationship('YHFinanceDto', back_populates='tradings')
    naver_finance = db.relationship('StockDto', back_populates='tradings')

    def __init__(self, email, kospi_stock_id, nasdaq_stock_id, stock_qty, price, trading_date):
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
        sql = cls.query.filter_by(cls.id == trading.id)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_email(cls, trading):
        sql = cls.query.filter_by(cls.email==trading.email)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(trading):
        db.session.add(trading)
        db.session.commit()

    @staticmethod
    def modify_trading(trading):
        Session = openSession()
        session = Session()
        trading = session.query(TradingDto)\
        .filter(TradingDto.id==trading.id)\
        .update({TradingDto.stock_qty: trading['stock_qty'], TradingDto.price: trading['price'], TradingDto.trading_date: trading['trading_date']})
        session.commit()
        session.close()

    @classmethod
    def delete_trading(cls, id):
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
parser.add_argument('kospi_stock_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('nasdaq_stock_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('stock_qty', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('price', type=float, required=True, help='This field cannot be left blank')
parser.add_argument('trading_date', type=str, required=True, help='This field cannot be left blank')


class Trading(Resource):        
        
    @staticmethod
    def post():
        body = request.get_json()
        print(f'body: {body}')
        trading = TradingDto(**body)
        TradingDao.save(trading)
        return {'trading': str(trading.id)}, 200
    
    @staticmethod
    def get(id):
        try:
            trading = TradingDao.find_by_id(id)
            if trading:
                return trading
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

    def put(self, id):
        args = parser.parse_args()
        print(f'Trading {args} updated')
        try:
            TradingDao.update(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

    @staticmethod
    def delete(id):
        try:
            TradingDao.delete(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

class Tradings(Resource):

    def post(self):
        t_dao = TradingDao()
        t_dao.insert_many('tradings')

    def get(self):
        data = TradingDao.find_all()
        return data, 200