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
from matplotlib import pyplot as plt



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class NewsDto(db.Model):
    __tablename__ = 'korea_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.Text)
    url :str = db.Column(db.String(500))
    ticker : str = db.Column(db.String(30))
    label : float = db.Column(db.Float)


    
    def __init__(self, id, date, headline, url, ticker, label):
        self.id = id
        self.date = date
        self.headline = headline
        self.url = url
        self.ticker = ticker
        self.label = label
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            url={self.url},ticker={self.ticker},label={self.label}'
            

    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'url':self.url,
            'ticker':self.ticker,
            'label':self.label
        }

class NewsVo:
    id : int = 0
    date: str =''
    headline: str=''
    url: str =''
    ticker: str =''
    label: float =0.0


Session = openSession()
session= Session()




class NewsDao(NewsDto):
    
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    

    def bulk(self):
        path = self.data
        companys = ['011070','051910']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            del df['Unnamed: 0']
            del df['content']
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
    def find_by_ticker(cls, tic):
        return session.query(NewsDto).filter(NewsDto.ticker.ilike(tic))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker'] == stock.ticker]
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_all(cls):
        sql =cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    

    @classmethod
    def find_by_id(cls,id):
        return session.query(NewsDto).filter(NewsDto.id.like(id)).one()

    @classmethod
    def find_by_date(cls,date):
        return session.query(NewsDto).filter(NewsDto.date.like(date)).one()

    @classmethod
    def find_by_headline(cls, headline):
        return session.query(NewsDto).filter(NewsDto.headline.like(headline)).one()
        
    @classmethod
    def find_by_content(cls,content):
        return session.query(NewsDto).filter(NewsDto.contet.like(content)).one()

    @classmethod
    def find_by_url(cls,url):
        return session.query(NewsDto).filter(NewsDto.url.like(url)).one()

    @classmethod
    def find_by_label(cls,label):
        return session.query(NewsDto).filter(NewsDto.label.like(label)).one()




# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('headline', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('url', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('label', type=float, required=True, help='This field cannot be left blank')


class News(Resource):

    @staticmethod
    def post(self):
        data = self.parser.parse_args()
        lnews = NewsDto(data['date'],data['headline'],data['content'],data['url'],data['ticker'],data['label'])
        try:
            lnews.save(data)
            return {'code':0, 'message':'SUCCESS'},200
        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return lnews.json(), 201
    
    
    @staticmethod
    def get(ticker):
        args = parser.parse_args()
        stock = NewsVo()
        stock.ticker = ticker
        data = NewsDao.find_all_by_ticker(stock)
        return data, 200

    def put(self, id):
        data = News.parser.parse_args()
        stock = NewsDao.find_by_id(id)

        stock.date = data['date']
        stock.ticker = data['ticker']
        stock.url = data['url']
        stock.headline = data['headline']
        stock.label=data['label']
        stock.save()
        return stock.json()

class News_(Resource):
    def get(self):
        return NewsDao.find_all(), 200
    
    # @staticmethod
    # def post():
    #     nd = NewsDao()
    #     nd.insert('naver_news')
    
    # @staticmethod
    # def get():
    #     data = NewsDao.find_all()
    #     return data, 200


class Lgchem_Label(Resource):
    
    @classmethod
    def get(cls):
        query = NewsDao.find_by_ticker('051910')
        df = pd.read_sql_query(query.statement, query.session.bind, parse_dates=['date'])
        means = df.resample('D', on='date').mean().dropna()
        print(means)
        means.insert(0, 'date', means.index)
        data = json.loads(means.to_json(orient='records'))
        #print(data)
        return data, 200

"""
                   id     label
date                            
2020-01-02  691.200000  0.444000
"""


class Lginnotek_Label(Resource):
    
    @classmethod
    def get(cls):
        query = NewsDao.find_by_ticker('011070')
        df = pd.read_sql_query(query.statement, query.session.bind, parse_dates=['date'])
        means = df.resample('D', on='date').mean().dropna()
        print(means)
        means.insert(0, 'date', means.index)
        data = json.loads(means.to_json(orient='records'))
        #print(data)
        return data, 200


    

# if __name__ =='__main__':
#     r = Lgchem_Label()
#     r.get()
