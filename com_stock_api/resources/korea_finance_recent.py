# -*- coding: utf-8 -*- 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import json
from mysql.connector.dbapi import Date
from sqlalchemy.dialects.mysql import DATE



class KoreaStock():
    
    def __init__(self):
        self.stock_code = None

    def new_model(self):
        print(f'ENTER STEP 1 : new_model ')
        stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                       header=0)[0]
        stock_code.종목코드=stock_code.종목코드.map('{:06d}'.format)
        stock_code=stock_code[['회사명','종목코드']]

        stock_code=stock_code.rename(columns={'회사명':'company','종목코드':'code'})
        
        #code_df.head()
        self.stock_code = stock_code
    
    def search_stock(self,company):
        print(f'ENTER STEP 2 : search_news ')
        print(f'company : {company}')

        result=[]

        stock_code = self.stock_code
        plusUrl = company.upper()
        plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()
        
        for i in range(1,15):
            url='https://finance.naver.com/item/sise_day.nhn?code='+str(plusUrl)+'&page={}'.format(i)
            response=requests.get(url)
            text=response.text
            html=BeautifulSoup(text,'html.parser')
            table0=html.find_all("tr",{"onmouseover":"mouseOver(this)"})
            #print(url)
            
            def refine_price(text):
                price=int(text.replace(",",""))
                return price

            
            for tr in table0:
                date= tr.find_all('td')[0].text
                temp=[]  
                
                for idx,td in enumerate(tr.find_all('td')[1:]):
                    if idx==1:
                        try:
                            #print(td.find()['alt'])
                            temp.append(td.find()['alt']) 
                        except: 
                            temp.append('')
                        
                    price=refine_price(td.text)
                    #print(price)
                    temp.append(price)
                
                #print([date]+temp)
                result.append([date]+temp)

                df_result=pd.DataFrame(result,columns=['date','close','up/down','pastday','open','high','low','volume'])
                df_result['ticker']=plusUrl 
                df_result.drop(['up/down', 'pastday'], axis='columns', inplace=True)
                #df_result['date']=pd.to_datetime(df_result['date'].astype(str), format='%Y/%m/%d')
                #print(df_result['date'])
                #df_result.set_index('date', inplace=True)
                
            return df_result
                
# if __name__ == "__main__":
#     ks = KoreaStock()
#     ks.new_model()
#     df_result = ks.search_stock('lg화학')
#     print(df_result)
    
    
class StockDto(db.Model):
    
    __tablename__ = 'korea_recent_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(DATE)
    open : int = db.Column(db.String(30))
    close : int = db.Column(db.String(30))
    high : int = db.Column(db.String(30))
    low :int = db.Column(db.String(30))
    volume : int = db.Column(db.String(30))
    ticker : str = db.Column(db.String(30))

    def __init__(self,id, date, open, close, high, low, volume, ticker):
        self.id = id
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.ticker = ticker
    
    def __repr__(self):
        return f'id={self.id}, date={self.date}, open={self.open},\
            close={self.close}, high={self.high}, low={self.low}, volume={self.volume}, ticker={self.ticker}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'ticker' : self.ticker
        }

class StockVo:
    id: int = 0
    date: str= ''
    open: int =''
    close: int =''
    high: int =''
    low: int =''
    volume: int =''
    ticker: str=''


Session = openSession()
session= Session()




class RecentStockDao(StockDto):

    # def __init__(self):
    #     self.data = os.path.abspath(__file__+"/.."+"/data/")

    @staticmethod
    def bulk():   #self
        krs = KoreaStock()
        krs.new_model()
        companys = ['lg화학','lg이노텍']
        for com in companys:
            df = krs.search_stock(com)
            #return df
            #path = self.data
            #df = pd.read_csv(path +'/lgchem.csv',encoding='utf-8',dtype=str)
            print(df.head())
            session.bulk_insert_mappings(StockDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(StockDto.id)).one()

    @staticmethod
    def save(stock):
        db.session.add(stock)
        db.sessio.commit()

    @staticmethod
    def update(stock):
        db.session.add(stock)
        db.session.commit()

    @classmethod
    def delete(cls,open):
        data = cls.query.get(open)
        db.session.delete(data)
        db.sessio.commit()
    
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_date(cls,date):
        return cls.query.filter_by(date == date).all()


    @classmethod
    def find_by_id(cls, open):
        return cls.query.filter_by(open == open).first()

    @classmethod
    def login(cls,stock):
        sql = cls.query.fillter(cls.id.like(stock.date)).fillter(cls.open.like(stock.open))
        
        df = pd.read_sql(sql.statement, sql.session.bind)
        print('----------')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    

if __name__ == "__main__":
    RecentStockDao.bulk()
    #s = RecentStockDao()
    #s.bulk()

    




# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================






parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('open',type=int, required=True,help='This field should be a open')
parser.add_argument('close',type=int, required=True,help='This field should be a close')
parser.add_argument('high',type=int, required=True,help='This field should be a high')
parser.add_argument('low',type=int, required=True,help='This field should be a low')
parser.add_argument('volume',type=int, required=True,help='This field should be a amount')
parser.add_argument('ticker',type=str, required=True,help='This field should be a stock')



class RecentStock(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Stock {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'

        params_str =''
        for key in params.keys():
            params_str += 'key: {}, value:{} <br>'.format(key,params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def get(id):
        print(f'Stock {id} added')
        try:
            stock = RecentStockDao.find_by_id(id)
            if stock:
                return stock.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'Stock {args["id"]} updated')
        return {'code':0, 'message': 'SUCCESS'}, 200

    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Stock {args["id"]} deleted')
        return {'code':0, 'message': 'SUCCESS'}, 200

class Stocks(Resource):
    
    @staticmethod
    def get():
        sd = RecentStockDao()
        sd.insert('korea_recent_finance')
    
    @staticmethod
    def get():
        data = RecentStockDao.find_all()
        return data, 200

class Auth(Resource):
    
    @staticmethod
    def post():
        body = request.get_json()
        stock = StockDto(**body)
        StockDto.save(stock)
        id = stock.id

        return {'id': str(id)}, 200

class Access(Resource):

    @staticmethod
    def post():
        args = parser.parse_argse()
        stock = StockVo()
        stock.id = args.id 
        sstock.date = args.date
        print(stock.id)
        print(stock.date)
        data = RecentStockDao.login(stock)
        return data[0], 200
