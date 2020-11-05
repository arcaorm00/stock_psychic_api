import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json

from com_stock_api.resources.korea_covid import KoreaDto
from com_stock_api.resources.korea_finance import StockDto
from com_stock_api.resources.korea_news import NewsDto



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    ticker : str = db.Column(db.VARCHAR(30))
    pred_price : int = db.Column(db.VARCHAR(30))

    covid_id: str = db.Column(db.Integer, db.ForeignKey(KoreaDto.id))
    stock_id: str = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    news_id: str = db.Column(db.Integer, db.ForeignKey(NewsDto.id))


    def __init__(self,id,date, covid_id,stock_id,news_id,ticker, pred_price):
        self.id = id
        self.date = date
        self.covid_id = covid_id
        self.stock_id= stock_id
        self.news_id= news_id
        self.ticker= ticker
        self.pred_price = pred_price
    
    def __repr__(self):
        return f'id={self.id},date={self.date},covid_id ={self.covid_id },stock_id={self.stock_id},news_id={self.news_id}, ticker={self.ticker},\
            pred_price={self.pred_price}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'covid_id': self.covid_id,
            'stock_id': self.stock_id,
            'news_id': self.news_id,
            'ticker' : self.ticker,
            'pred_price' : self.pred_price
        }

class KospiVo:
    id : int = 0
    date : str =''
    ticker: str =''
    pred_price : int = 0
    covid_id : str = 0
    stock_id : str = 0
    news_id : str =  0


Session = openSession()
session= Session()


class KospiDao(KospiDto):

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")

    def bulk(self):
        path = self.data
        companys = ['lgchem','lginnotek']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            df = df.iloc[84:206, : ]           
            session.bulk_insert_mappings(KospiDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(KospiDto.id)).one()
    
    @staticmethod
    def save(data):
        db.session.add(data)
        db.session.commit()

    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()

    @classmethod
    def delete(cls,id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()


    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker'] == stock.ticker]
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls,id):
        return session.query(KospiDto).filter(KospiDto.id.like(id)).one()
        
    @classmethod
    def find_by_date(cls, date):
        return session.query(KospiDto).filter(KospiDto.date.like(date)).one()

    @classmethod
    def find_by_predprice(cls,pred_price):
        return session.query(KospiDto).filter(KospiDao.pred_price.like(pred_price)).one()

    @classmethod
    def find_by_stockid(cls,stock_id):
        return session.query(KospiDto).filter(KospiDto.stock_id.like(stock_id)).one()

    @classmethod
    def find_by_ticker(cls,ticker):
        return session.query(KospiDto).filter(KospiDto.ticker.like(ticker)).one()

    @classmethod
    def find_by_covidid(cls,covid_id):
        return session.query(KospiDto).filter(KospiDto.covid_id.like(covid_id)).one()

    @classmethod
    def fidn_by_newsid(cls,news_id):
        return session.queryfilter(KospiDto.news_id.like(news_id)).one()


# if __name__ =='__main__':
#     #KospiDao()
#     r=KospiDao()
#     r.bulk()

# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('covid_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('stock_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('news_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('pred_price', type=int, required=True, help='This field cannot be left blank')

class Kospi(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        kospiprediction = KospiDto(data['date'], data['ticker'],data['pred_price'], data['stock_id'], data['covid_id'], data['news_id'])
        try:
            kospiprediction.save(data)
            return {'code':0, 'message':'SUCCESS'},200
        except:
            return {'message': 'An error occured inserting the pred history'}, 500
        return kospiprediction.json(),201

    def get(self, id):
        kospiprediction = KospiDao.find_by_id(id)
        if kospiprediction:
            return kospiprediction.json()
        return {'message': 'kospiprediction not found'}, 404
    
    def put(self, id):
        data = Kospi.parser.parse_args()
        prediction = KospiDao.find_by_id(id)

        prediction.date = data['date']
        prediction.price = data['pred_price']
        prediction.save()
        return prediction.json()

class Kospis(Resource):
    def get(self):
        return  KospiDao.find_all(), 200

class lgchem_pred(Resource):
    @staticmethod
    def get():
        stock = KospiVo()
        stock.ticker='051910'
        data = KospiDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def post():
        args = parser.parse_args()
        stock = KospiVo()
        stock.ticker = args.ticker
        data = KospiDao.find_all_by_ticker(stock)
        return data[0], 200

class lginnotek_pred(Resource):

    @staticmethod
    def get():
        stock = KospiVo()
        stock.ticker='011070'
        data = KospiDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def post():
        args = parser.parse_args()
        stock = KospiVo()
        stock.ticker = args.ticker
        data = KospiDao.find_all_by_ticker(stock)
        return data[0], 200



        
