from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.member import MemberDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from com_stock_api.resources.korea_finance import StockDto
import datetime

import pandas as pd
import numpy as np
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse

import random
from sqlalchemy import func

import datetime



'''
 * @ Module Name : trading.py
 * @ Description : Trading stock
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     flask_restful
 * @ 수정일         수정자                      수정내용
 *   ------------------------------------------------------------------------
 *   2020.10.31     곽아름      데이터베이스에 insert할 자동 매수 class 추가
''' 





# =====================================================================
# =====================================================================
# =================        preprocessing         ======================
# =====================================================================
# =====================================================================





class TradingPro:

    def __init__(self):
        self.members = object
        self.kospis = object
        self.nasdaqs = object

    def hook(self):
        self.get_data()
        data = self.make_data()
        return data

    def get_data(self):
        members = pd.read_sql_table('members', engine.connect())
        kospis = pd.read_sql_table('korea_finance', engine.connect())
        nasdaqs = pd.read_sql_table('yahoo_finance', engine.connect())
        
        # kospi의 금액을 모두 20201030 현재 환율 1129.16으로 나눔
        kospis['open'] = [round(float(k)/1129.16, 4) for k in kospis['open']]
        kospis['close'] = [round(float(k)/1129.16, 4) for k in kospis['close']]
        kospis['high'] = [round(float(k)/1129.16, 4) for k in kospis['high']]
        kospis['low'] = [round(float(k)/1129.16, 4) for k in kospis['low']]

        self.members = members
        self.kospis = kospis
        self.nasdaqs = nasdaqs
    
    def make_data(self):
        kospi_ticker = self.kospis['ticker'].unique()
        nasdaq_ticker = self.nasdaqs['ticker'].unique()

        # tickers = ['051910' '011070' 'AAPL' 'TSLA']
        tickers = np.hstack([kospi_ticker, nasdaq_ticker])
        tickers[np.where(tickers == '051910')] = 'LG화학'
        tickers[np.where(tickers == '011070')] = 'LG이노텍'
        # tickers = ['LG화학' 'LG이노텍' 'AAPL' 'TSLA']

        trading_arr = []
        email = ''
        stock_type = ''
        stock_ticker = ''
        stock_qty = 0
        price = 0
        trading_date = ''

        for idx, member in self.members.iterrows():
            members_trading_qty = int(member['stock_qty'])

            if int(member['balance']) <= 0: continue
            
            # tickers의 값 중 회원의 stock_qty 수만큼 랜덤 추출
            random_ticker = random.choices(tickers, k=members_trading_qty)

            for tick in random_ticker:
                email = member['email']
                stock_ticker = tick
                
                if tick == 'LG화학': tick = '051910'
                if tick == 'LG이노텍': tick = '011070'

                if (self.nasdaqs['ticker'] == tick).any():
                    stock_type = 'NASDAQ'
                    nasdaq = self.nasdaqs[self.nasdaqs['ticker'] == tick]
                    
                    trading_date = ''
                    while True:
                        trading_date = random.choice(self.nasdaqs['date'])
                        if trading_date > datetime.datetime.strptime('2017-02-28', '%Y-%m-%d'):
                            break
                    high = float(nasdaq[nasdaq['date'] == trading_date]['high'])
                    low = float(nasdaq[nasdaq['date'] == trading_date]['low'])
                    price = round(random.uniform(high, low), 4)

                    stock_qty = (float(member['balance'])/members_trading_qty)/price

                if (self.kospis['ticker'] == tick).any():
                    stock_type = 'KOSPI'
                    kospi = self.kospis[self.kospis['ticker'] == tick]

                    trading_date = random.choice(self.kospis['date'])
                    high = float(kospi[kospi['date'] == trading_date]['high'])
                    low = float(kospi[kospi['date'] == trading_date]['low'])
                    price = round(random.uniform(high, low), 4)

                    stock_qty = int((float(member['balance'])/members_trading_qty)/price)
                
                result = {'email': email, 'stock_type': stock_type, 'stock_ticker': stock_ticker, 'stock_qty': int(stock_qty), 'price': price, 'trading_date': str(trading_date)}
                trading_arr.append(result)
       
        trading_df = pd.DataFrame(trading_arr)
        return trading_df




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
    stock_type: str = db.Column(db.String(20), nullable=False)
    stock_ticker: str = db.Column(db.String(50), nullable=False)
    stock_qty: int = db.Column(db.Integer, nullable=False)
    price: float = db.Column(db.FLOAT, nullable=False)
    trading_date: str = db.Column(db.String(50), default=datetime.datetime.now())

    member = db.relationship('MemberDto', back_populates='tradings')

    def __init__(self, email, stock_type, stock_ticker, stock_qty, price, trading_date):
        self.email = email
        self.stock_type = stock_type
        self.stock_ticker = stock_ticker
        self.stock_qty = stock_qty
        self.price = price
        self.trading_date = trading_date

    def __repr__(self):
        return 'Trading(trading_id={}, email={}, stock_type={}, stock_ticker={}, stock_qty={}, price={}, trading_date={})'.format(self.id, self.email, self.stock_type, self.stock_ticker, self.stock_qty, self.price, self.trading_date)
    
    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_ticker': self.stock_ticker,
            'stock_qty': self.stock_qty,
            'price': self.price,
            'trading_date': self.trading_date
        }

class TradingVo:
    id: int = 0
    email: str = ''
    stock_type: int = 0
    stock_ticker: int = 0
    stock_qty: int = 0
    price: float = 0.0
    trading_date: str = db.Column(db.String(1000), default=datetime.datetime.now())




Session = openSession()
session = Session()

class TradingDao(TradingDto):

    def __init__(self):
        ...

    @classmethod
    def count(cls):
        return session.query(func.count(TradingDto.id)).one()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, trading):
        sql = cls.query.filter(cls.id == trading.id)
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
    def save(trading):
        db.session.add(trading)
        db.session.commit()
        session.close()

    @staticmethod
    def insert_many():
        service = TradingPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(TradingDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

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
parser.add_argument('stock_type', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('stock_ticker', type=int, required=False, help='This field cannot be left blank')
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

    @staticmethod
    def put(id):
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

    def get(self, email):
        print(email)
        data = TradingDao.find_by_email(email)
        return data, 200