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

# from com_stock_api.resource.korea_covid import KoreaDto
# from com_stock_api.resource.korea_finance import StockDto
# from com_stock_api.resource.korea_news import NewsDto


class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    ticker : str = db.Column(db.VARCHAR(30))
    pred_price : int = db.Column(db.VARCHAR(30))

    # covid_id: str = db.Column(db.Integer, db.ForeignKey(KoreaDto.id))
    # stock_id: str = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    # news_id: str = db.Column(db.Integer, db.ForeignKey(NewsDto.id))


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
    pred_price : int =''
    covid_id : str =''
    stock_id : str =''
    news_id : str ='' 


Session = openSession()
session= Session()


class KospiDao(KospiDto):

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    
    #@staticmethod
    def bulk(self):
        #service = KospiService()
        #df = service.hook()
        path = self.data
        df = pd.read_csv(path+'/movie_review.csv',encoding='utf-8', dtype=str)
        session.bulk_insert_mappings(KospiDto, df.to_dict(orient='records'))
        session.commit()
        session.close()
    
    @staticmethod
    def conut():
        return session.query(func.count(KospiDto.id)).one()
    
    @staticmethod
    def save(kospi):
        db.session.add(kospi)
        db.session.commit()

    @staticmethod
    def update(kospi):
        db.session.add(kospi)
        db.session.commit()

    @classmethod
    def delete_kospi(cls,date):
        data = cls.query.get(date)
        db.session.delete(data)
        db.session.commit()


    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls,id):
        return cls.query.filter_by(id == id).all()


    @classmethod
    def find_by_date(cls, date):
        return cls.query.filter_by(date == date).first()

    @classmethod
    def login(cls,kospi):
        sql = cls.query.fillter(cls.id.like(kospi.id)).fillter(cls.date.like(kospi.date))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('==================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_sjon(orient='records'))

if __name__ =="__main__":
    #KospiDao.bulk()
    kp = KospiDao()
    kp.bulk()


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================



parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field should be a id')
parser.add_argument('date', type=str, required=True, help='TThis field should be a date')
parser.add_argument('covid_id', type=int, required=True, help='This field should be a covid_date')
parser.add_argument('stock_id', type=int, required=True, help='This field should be a stock_date')
parser.add_argument('news_id', type=int, required=True, help='This field should be a news_date')
parser.add_argument('ticker', type=str, required=True, help='This field should be a ticker')
parser.add_argument('pred_price', type=int, required=True, help='This field should be a price')

class Kospi(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Kospi {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'
        params_str=''
        for key in params.keys():
            params_str += 'key:{},value:{}<br>'.format(key, params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def post(id):
        print(f'Kospi{id} added')
        try:
            kospi = KospiDao.find_by_id(id)
            if kospi:
                return kospi.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
            
    @staticmethod
    def update():
            args = parser.arse_args()
            print(f'Kospi {args["id"]} updated')
            return {'code':0, 'message':'SUCCESS'}, 200
            
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Kospi {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class Kospis(Resource):

    @staticmethod
    def get():
        kd = KospiDao()
        kd.insert_many('kospi_pred')

    def get():
        data = KospiDao.find_all()
        return data, 200

class Auth(Resource):

    @staticmethod
    def post():
        body = request.get_json()
        kospi = KospiDto(**body)
        KospiDao.save(kospi)
        id = kospi.id

        return {'id': str(id)}, 200

class Access(Resource):
    
    @staticmethod
    def post():
        args = parser.parse_args()
        kospi = KospiVo()
        kospi.id = args.id
        kospi.date = args.date
        print(kospi.id)
        print(kospi.date)
        data = KospiDao.login(kospi)
        return data[0], 200