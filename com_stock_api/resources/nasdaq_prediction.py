from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.resources.uscovid import USCovidDto
from com_stock_api.resources.investingnews import InvestingDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from sqlalchemy import and_,or_,func, extract
import os
import pandas as pd
import json
from datetime import datetime
from sklearn import preprocessing
import numpy as np

# =============================================================
# =============================================================
# =================      Deep Learning    =====================
# =============================================================
# =============================================================

class NasdaqPreprocessing():

    history_points: int = 50

    def dataset():
        ...




# =============================================================
# =============================================================
# ===================      Modeling    ========================
# =============================================================
# =============================================================

class NasdaqPredictionDto(db.Model):
    __tablename__ = 'NASDAQ_prediction'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    ticker: str = db.Column(db.String(30))
    date: str = db.Column(db.Date)
    pred_price: float = db.Column(db.Float)
    
    stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    covid_id : int = db.Column(db.Integer, db.ForeignKey(USCovidDto.id))
    news_id: int = db.Column(db.Integer, db.ForeignKey(InvestingDto.id))


    def __init__(self, ticker, date, pred_price, stock_id, covid_id, news_id):
        self.ticker = ticker
        self.date = date
        self.pred_price = pred_price

        self.stock_id = stock_id
        self.covid_id = covid_id
        self.news_id = news_id

    def __repr__(self):
        return f'NASDAQ_Prediction(id=\'{self.id}\',ticker=\'{self.ticker}\',date=\'{self.date}\',\
                pred_price=\'{self.pred_price}\',stock_id=\'{self.stock_id}\',\
                covid_id=\'{self.covid_id}\', news_id=\'{self.news_id}\' )'

    @property
    def json(self):
        return {
            'id' : self.id,
            'ticker' : self.ticker,
            'date' : self.date,
            'pred_price' : self.pred_price,
            'stock_id' : self.stock_id,
            'covid_id' : self.covid_id,
            'news_id' : self.news_id
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commint()

class NasdaqPredictionVo:
    id: int = 0
    ticker: str = ''
    date : str = ''
    pred_price: float = 0.0
    stock_id : str = ''
    covid_id : str = ''
    news_id : str = ''


Session = openSession()
session = Session()


class NasdaqPredictionDao(NasdaqPredictionDto):

    @staticmethod
    def count():
        return session.query(func.count(NasdaqPredictionDto.id)).one()

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


    @staticmethod   
    def bulk():
        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/data/")
            file_name = tic + '_pred.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(NasdaqPredictionDto, df.to_dict(orient="records"))
            session.commit()
        session.close()

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']==stock.ticker]
        return json.loads(df.to_json(orient='records'))

    
    @classmethod
    def find_by_date(cls, date, tic):
        return session.query.filter(and_(cls.date.like(date), cls.ticker.ilike(tic)))
    @classmethod
    def find_by_ticker(cls, tic):
        print("In find_by_ticker")
   
        return session.query(NasdaqPredictionDto).filter((NasdaqPredictionDto.ticker.ilike(tic))).order_by(NasdaqPredictionDto.date).all()
        
    @classmethod
    def find_by_period(cls,tic, start_date, end_date):
        return session.query(NasdaqPredictionDto).filter(and_(NasdaqPredictionDto.ticker.ilike(tic),date__range=(start_date, end_date)))
    @classmethod
    def find_today_one(cls, tic):
        today = datetime.today()
        return session.query(NasdaqPredictionDto).filter(and_(NasdaqPredictionDto.ticker.ilike(tic),NasdaqPredictionDto.date.like(today)))


# =============================================================
# =============================================================
# ===================      Resourcing    ======================
# =============================================================
# =============================================================

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('pred_price', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('stock_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('covid_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('news_id', type=int, required=False, help='This field cannot be left blank')

class NasdaqPrediction(Resource):    
    @staticmethod
    def post():
        data = parser.parse_args()
        nprediction = NasdaqPredictionDto(data['date'], data['ticker'],data['pred_price'], data['stock_id'], data['covid_id'], data['news_id'])
        try: 
            nprediction.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200
        except:
            return {'message': 'An error occured inserting the article'}, 500
        return nprediction.json(), 201
    
    
    def get(self, id):
        article = NasdaqPredictionDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404

    def put(self, id):
        data = NasdaqPrediction.parser.parse_args()
        prediction = NasdaqPredictionDao.find_by_id(id)

        prediction.title = data['title']
        prediction.content = data['content']
        prediction.save()
        return prediction.json()

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Ticker {args["ticker"]} on date {args["date"]} is deleted')
        NasdaqPredictionDao.delete(args['id'])
        return {'code' : 0 , 'message' : 'SUCCESS'}, 200

class NasdaqPredictions(Resource):
    def get(self):
        return NasdaqPredictionDao.find_all(), 200
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}

class TeslaPredGraph(Resource):

    @staticmethod
    def get():
        print("=====nasdaq_prediction.py / TeslPredaGraph's get")
        stock = NasdaqPredictionVo
        stock.ticker = 'TSLA'
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====nasdaq_prediction.py / TeslaPredGraph's post")
        args = parser.parse_args()
        stock = NasdaqPredictionVo
        stock.ticker = args.ticker
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data[0], 200
        
class ApplePredGraph(Resource):

    @staticmethod
    def get():
        print("=====nasdaq_prediction.py / ApplePredGraph's get")
        stock = NasdaqPredictionVo
        stock.ticker = 'AAPL'
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====nasdaq_prediction.py / ApplePredGraph's post")
        args = parser.parse_args()
        stock = NasdaqPredictionVo()
        stock.ticker = args.ticker
        print("TICKER: " , stock.ticker)
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data[0], 200