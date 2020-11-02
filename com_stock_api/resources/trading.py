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
    '''
    이 클래스가 해야할 일
        1) yahoo_finance와 korea_finance 테이블의 정보 중 date가 2020으로 시작하는 high와 low 사이의 금액을 랜덤으로 구한다. (tradings: price)
        2) 해당 date는 tradings의 trading_date가 된다. (tradings: trading_date)
        3) yahoo_finance의 종목을 거래했다면 NASDAQ, korea_finance의 종목을 거래했다면 KOSPI (tradings: stock_type)
        4) 회원의 balance / price 값의 소수점 버림이 보유주의 수 (tradings: stock_qty)
        5) 회원의 stock_qty 만큼 새로운 종목을 거래해야 함 (ex. 회원 stock_qty가 2라면 TSLA와 LG화학)
        !!! yahoo finance와 korea finance의 환율이 다르다! 달러 기준으로 변경해서 계산해야함 !!!
    '''

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

        '''
        MEMBERS TABLE:                 email password      name      profile geography  gender  ...  credit_score  is_active_member  estimated_salary       role  probability_churn  exited
        0  15565701@gmail.com     1234     Ferri  noimage.png     Spain  Female  ...           698                 0           90212.4  ROLE_USER               -1.0       0     
        1  15565706@gmail.com     1234  Akobundu  noimage.png     Spain    Male  ...           612                 1           83256.3  ROLE_USER               -1.0       1     
        2  15565714@gmail.com     1234  Cattaneo  noimage.png    France    Male  ...           601                 1           96518.0  ROLE_USER               -1.0       0     
        3  15565779@gmail.com     1234      Kent  noimage.png   Germany  Female  ...           627                 0          188258.0  ROLE_USER               -1.0       0     
        4  15565796@gmail.com     1234  Docherty  noimage.png   Germany    Male  ...           745                 0           74510.6  ROLE_USER               -1.0       0     
        [5 rows x 17 columns]
        KOSPI TABLE:    id       date    open   close    high     low  volume  ticker
        0   1 2020-10-16  633000  640000  643000  628000  309530  051910
        1   2 2020-10-15  636000  637000  648000  629000  531454  051910
        2   3 2020-10-14  642000  628000  644000  620000  725349  051910
        3   4 2020-10-13  675000  644000  678000  640000  678451  051910
        4   5 2020-10-12  680000  672000  692000  670000  551057  051910
        NASDAQ TABLE:    id ticker       date     open    high      low    close  adjclose     volume
        0   1   AAPL 2020-01-02  74.0600  75.150  73.7975  75.0875   74.5730  135480400
        1   2   AAPL 2020-01-03  74.2875  75.145  74.1250  74.3575   73.8480  146322800
        2   3   AAPL 2020-01-06  73.4475  74.990  73.1875  74.9500   74.4365  118387200
        3   4   AAPL 2020-01-07  74.9600  75.225  74.3700  74.5975   74.0864  108872000
        4   5   AAPL 2020-01-08  74.2900  76.110  74.2900  75.7975   75.2782  132079200
        '''
        
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

                    trading_date = random.choice(self.nasdaqs['date'])
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
        print(trading_df[:50])
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
    # kospi_stock_id: int = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    # nasdaq_stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    stock_type: str = db.Column(db.String(20), nullable=False)
    stock_ticker: str = db.Column(db.String(50), nullable=False)
    stock_qty: int = db.Column(db.Integer, nullable=False)
    price: float = db.Column(db.FLOAT, nullable=False)
    trading_date: str = db.Column(db.DATE, default=datetime.datetime.now())

    member = db.relationship('MemberDto', back_populates='tradings')
    # yahoo_finance = db.relationship('YHFinanceDto', back_populates='tradings')
    # korea_finance = db.relationship('StockDto', back_populates='tradings')

    def __init__(self, email, stock_type, stock_ticker, stock_qty, price, trading_date):
        self.email = email
        self.stock_type = stock_type
        self.stock_ticker = stock_ticker
        self.stock_qty = stock_qty
        self.price = price
        self.trading_date = trading_date
    # def __init__(self, email, kospi_stock_id, nasdaq_stock_id, stock_qty, price, trading_date):
    #     self.email = email
    #     self.kospi_stock_id = kospi_stock_id
    #     self.nasdaq_stock_id = nasdaq_stock_id
    #     self.stock_qty = stock_qty
    #     self.price = price
    #     self.trading_date = trading_date

    def __repr__(self):
        return 'Trading(trading_id={}, member_id={}, stock_type={}, stock_ticker={}, stock_qty={}, price={}, trading_date={})'.format(self.id, self.member_id, self.stock_type, self.stock_ticker, self.stock_qty, self.price, self.trading_date)
    
    @property
    def json(self):
        return {
            'id': self.id,
            'member_id': self.member_id,
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