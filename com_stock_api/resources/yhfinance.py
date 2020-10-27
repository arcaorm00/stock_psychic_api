from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import os

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
    volume : int = db.Column(db.Integer)
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
                adjclose=\'{self.adjclose}\',volume=\'{self.volume}\',)'


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

class YHFinanceDao(YHFinanceDto):

    @classmethod
    def count(cls):
        return cls.query.count()
        
    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_date(cls, date):
        return cls.query.filer_by(date == date).all()
    
    @staticmethod   
    def insert_many():
        Session = openSession()
        session = Session()
        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/data/")
            file_name = tic + '.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(YHFinanceDto, df.to_dict(orient="records"))
            session.commit()
        session.close()


# =============================================================
# =============================================================
# ======================      CONTROLLER    ======================
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
    def post(self):
        data = self.parset.parse_args()
        stock = YHFinanceDto(data['date'], data['ticker'],data['open'], data['high'], data['low'], data['close'],  data['adjclose'], data['volume'])
        try: 
            stock.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting the covid case'}, 500
        return stock.json(), 201
        
    def get(self, id):
        stock = YHFinanceDao.find_by_id(id)
        if stock:
            return stock.json()
        return {'message': 'The stock was not found'}, 404

    def put(self, id):
        data = YHFinance.parser.parse_args()
        stock = YHFinanceDao.find_by_id(id)

        stock.title = data['title']
        stock.content = data['content']
        stock.save()
        return stock.json()

class YHFinances(Resource):
    def get(self):
        return {'stock history': list(map(lambda article: article.json(), YHFinanceDao.find_all()))}
