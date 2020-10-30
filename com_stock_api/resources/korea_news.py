import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from com_stock_api.utils.checker import is_number
from collections import defaultdict
import numpy as np
import math
from pandas import read_table
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json


class NewsDto(db.Model):
    __tablename__ = 'korea_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    content : str = db.Column(db.Text) #String(10000)
    url :str = db.Column(db.String(500))
    ticker : str = db.Column(db.String(30))
    label : float = db.Column(db.Float)


    
    def __init__(self, id, date, headline, content, url, ticker, label):
        self.id = id
        self.date = date
        self.headline = headline
        self.content = content
        self.url = url
        self.ticker = ticker
        self.label = label
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            content={self.content},url={self.url},ticker={self.ticker},label={self.label}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'content':self.ccontent,
            'url':self.url,
            'ticker':self.ticker,
            'label':self.label
        }

class NewsVo:
    id : int = 0
    date: str =''
    headline: str=''
    content: str=''
    url: str =''
    ticker: str =''
    label: float =''


Session = openSession()
session= Session()




class NewsDao(NewsDto):
    
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    
    #@staticmethod
    def bulk(self):
        #service = NewsService()
        #df = service.hook()
        path = self.data
        companys = ['011070','051910']
        for com in companys:
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
        print(df.head())
        session.bulk_insert_mappings(NewsDto, df.to_dict(orient='records'))
        session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(NewsDto.id)).one()

    @staticmethod
    def save(news):
        db.session.add(news)
        db.session.commit()
    
    @staticmethod
    def update(news):
        db.session.add(news)
        db.session.commit()
    
    @classmethod
    def delete(cls,headline):
        data = cls.qeury.get(headline)
        db.session.delete(data)
        db.session.commit()
    
    @classmethod
    def find_all(cls):
        sql =cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    

    @classmethod
    def find_by_id(cls,id):
        return cls.query.filter_by(id == id).all()


    @classmethod
    def find_by_headline(cls, headline):
        return cls.query.filter_by(headline == headline).first()

    @classmethod
    def login(cls,news):
        sql = cls.query.fillter(cls.id.like(news.id)).fillter(cls.headline.like(news.headline))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('============================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))



if __name__ == "__main__":
    #NewsDao.bulk()
    n = NewsDao()
    n.bulk()


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field should be a id')
parser.add_argument('date', type=str, required=True, help='This field should be a date')
parser.add_argument('headline', type=str, required=True, help='This field should be a headline')
parser.add_argument('content', type=str, required=True, help='This field should be a content')
parser.add_argument('url', type=str, required=True, help='This field should be a url')
parser.add_argument('ticker', type=str, required=True, help='This field should be a stock')
parser.add_argument('label', type=float, required=True, help='This field should be a label')


class News(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'News {args["id"]} added')
        parmas = json.loads(request.get_data(), encoding='utf-8')
        if len (parmas) == 0:
            return 'No parameter'
        
        params_str=''
        for key in parmas.keys():
            params_str += 'key:{}, value:{}<br>'.format(key, parmas[key])
        return {'code':0, 'message': 'SUCCESS'}, 200
    
    @staticmethod
    def get(id):
        print(f'News {id} added')
        try:
            news = NewsDao.find_by_id(id)
            if news:
                return news.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'News {args["id"]} updated')
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'News {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class News_(Resource):
    
    @staticmethod
    def get():
        nd = NewsDao()
        nd.insert('naver_news')
    
    @staticmethod
    def get():
        data = NewsDao.find_all()
        return data, 200

class Auth(Resource):
    @staticmethod
    def post():
        body = request.get_json()
        news = NewsDto(**body)
        NewsDao.save(news)
        id = news.id

        return {'id': str(id)}, 200

class Access(Resource):
    
    @staticmethod
    def post():
        args = parser.parse_args()
        news = NewsVo()
        news.id = args.id
        news.headline = args.headline
        print(news.id)
        print(news.headline)
        data = NewsDao.login(news)
        return data[0], 200