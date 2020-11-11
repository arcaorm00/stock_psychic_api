from flask_restful import Resource, reqparse
from flask import request, jsonify
from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.serializer import loads, dumps
from sqlalchemy.orm import class_mapper
import pandas as pd
import os
import json
from sqlalchemy import and_,or_,func
from datetime import datetime, date
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 

# =============================================================
# =============================================================
# ================    DATA MINING && SERVICE    ===============
# =============================================================
# =============================================================

class YHFinancePro:
    tickers : str = ['AAPL', 'TSLA']
    ticker : str
    START_DATE: str = '2020-07-01'
    END_DATE: str = datetime.now()

    def __init__(self):
        self.ticker = ''
        
    def hook(self):
        histories = []
        for t in self.tickers:
            self.ticker = t
            history = self.saved_to_csv(self.get_history())
            histories.append(self.process_dataframe(history))
        return histories

    def get_history(self):
        data = pdr.get_data_yahoo(self.ticker, period='max')
        return data

    def get_history_by_date(self, start, end):
        data = pdr.get_data_yahoo(self.ticker, start=start, end=end)
        return data
    
    def get_file_path(self):
        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = self.ticker + '.csv'
        return os.path.join(path,file_name)

    def process_dataframe(self, df):
        input_file = self.get_file_path()
        print("input file: ", input_file)
        data = pd.read_csv(input_file)
        data.rename(columns = {'Date' : 'date', 'Open':'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close' :'adjclose', 'Volume':'volume'}, inplace = True)
        data.insert(loc=0, column='ticker', value=self.ticker)
        
        output_file = self.get_file_path()
        data.to_csv(output_file)
        return data

    def saved_to_csv(self, data):
        output_file = self.get_file_path()
        data.to_csv(output_file)

# =============================================================
# =============================================================
# =========================   Modeling   ======================
# =============================================================
# =============================================================


#주식 DB만들기
class YHFinanceDto(db.Model):
    __tablename__ = 'yahoo_finance'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    ticker : str = db.Column(db.String(10))
    date : str  = db.Column(db.Date)
    open : float = db.Column(db.Float)
    high : float = db.Column(db.Float)
    low : float = db.Column(db.Float)
    close : float = db.Column(db.Float)
    adjclose : float = db.Column(db.Float)
    volume : int = db.Column(db.BigInteger)
    #date format : YYYY-MM-DD
    # amount : unit = million 
    
    def __init__(self, ticker, date, open, high, low, close, adjclose, volume):
        self.ticker = ticker
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume

    # Date,Open,High,Low,Close,Adj Close,Volume
    def __repr__(self):
        return f'YHFinance(id=\'{self.id}\',ticker=\'{self.ticker}\', date=\'{self.date}\',open=\'{self.open}\', \
            high=\'{self.high}\',low=\'{self.low}\', close=\'{self.close}\',\
                adjclose=\'{self.adjclose}\',volume=\'{self.volume}\')'


    @property
    def json(self):
        return {
            'id' : self.id,
            'ticker' : self.ticker,
            'date' : self.date,
            'open' : self.open,
            'high' : self.high,
            'low' : self.low,
            'close' : self.close,
            'adjclose' : self.adjclose,
            'volume' : self.volume
        }

class YHFinanceVo:
    id: int = 0
    ticker: str = ''
    date : str = ''
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    adjclose: float = 0.0
    volume: int = 0

Session = openSession()
session = Session()


class YHFinanceDao(YHFinanceDto):

    @staticmethod
    def count():
        return session.query(func.count(YHFinanceDto.id)).one()

    @staticmethod
    def save(data):
        db.session.add(data)
        db.session.commit()
    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()

    @staticmethod
    def delete(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        
    @staticmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']==stock.ticker]
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def bulk():
        service = YHFinancePro()
        dfs = service.hook()
        for i in dfs:
            print(i.head())
            session.bulk_insert_mappings(YHFinanceDto, i.to_dict(orient="records"))
            session.commit()
        session.close()

    @classmethod
    def find_by_date(cls, date, tic):
        return session.query.filter(and_(cls.date.like(date), cls.ticker.ilike(tic)))
    @classmethod
    def find_by_ticker(cls, tic):   
        return session.query(YHFinanceDto).filter((YHFinanceDto.ticker.ilike(tic))).order_by(YHFinanceDto.date).all()
        
    @classmethod
    def find_by_period(cls,tic, start_date, end_date):
        return session.query(YHFinanceDto).filter(and_(YHFinanceDto.ticker.ilike(tic),date__range=(start_date, end_date)))
    @classmethod
    def find_today_one(cls, tic):
        today = datetime.today()
        return session.query(YHFinanceDto).filter(and_(YHFinanceDto.ticker.ilike(tic),YHFinanceDto.date.like(today)))

# =============================================================
# =============================================================
# =======================   Resourcing   ======================
# =============================================================
# =============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('open', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('high', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('low', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('close', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('adjclose', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('volume', type=int, required=False, help='This field cannot be left blank')
    

class YHFinance(Resource):

# Date,Open,High,Low,Close,Adj Close,Volume

    @staticmethod
    def post():
        data = self.parset.parse_args()
        stock = YHFinanceDto(data['date'], data['ticker'],data['open'], data['high'], data['low'], data['close'],  data['adjclose'], data['volume'])
        try: 
            stock.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting the stock history'}, 500
        return stock.json(), 201
        
    def get(ticker):
        stock = YHFinanceDao.find_by_ticker(ticker)
        if stock:
            return stock.json()
        return {'message': 'The stock was not found'}, 404

    def put(id):
        data = YHFinance.parser.parse_args()
        stock = YHFinanceDao.find_by_id(id)

        stock.date = data['date']
        stock.close = data['close']
        stock.save()
        return stock.json()

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Ticker {args["ticker"]} on date {args["date"]} is deleted')
        YHFinanceDao.delete(args['id'])
        return {'code' : 0 , 'message' : 'SUCCESS'}, 200

class YHFinances(Resource):
    def get(self):
        return YHFinanceDao.find_all(), 200

class TeslaGraph(Resource):

    @staticmethod
    def get():
        print("=====yhfinance.py / TeslaGraph's get")
        stock = YHFinanceVo()
        stock.ticker = 'TSLA'
        data = YHFinanceDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====yhfinance.py / TeslaGraph's post")
        args = parser.parse_args()
        stock = YHFinanceVo()
        stock.ticker = args.ticker
        data = YHFinanceDao.find_all_by_ticker(stock)
        return data[0], 200
        
        
class AppleGraph(Resource):

    @staticmethod
    def get():
        print("=====yhfinance.py / AppleGraph's get")
        stock = YHFinanceVo
        stock.ticker = 'AAPL'
        data = YHFinanceDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====yhfinance.py / AppleGraph's post")
        args = parser.parse_args()
        stock = YHFinanceVo()
        stock.ticker = args.ticker
        print("TICKER: " , stock.ticker)
        data = YHFinanceDao.find_all_by_ticker(stock)
        return data[0], 200