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
from com_stock_api.utils.checker import is_number
from collections import defaultdict
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import json
from sqlalchemy.dialects.mysql import DATE
import time
import random

# ==============================================================
# =========================                =====================
# =========================  Data Mining   =====================
# =========================                =====================
# ==============================================================


class KoreaNews():
    
    def __init__(self):
        self.stock_code = None

    def new_model(self):
        stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                       header=0)[0]
        stock_code.종목코드=stock_code.종목코드.map('{:06d}'.format)
        stock_code=stock_code[['회사명','종목코드']]

        stock_code=stock_code.rename(columns={'회사명':'company','종목코드':'code'})
        #code_df.head()
        self.stock_code = stock_code

    def search_news(self,company):
        article_result =[]
        title_result = []
        link_result = []
        date_result = []

        stock_code = self.stock_code
        plusUrl = company.upper()
        plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()
        
        
       
        for i in range(1,10):

            url = 'https://finance.naver.com/item/news_news.nhn?code='+ str(plusUrl)+'&page={}'.format(i)
            #print(f'url : {url}')
            source_code = requests.get(url).text
            html = BeautifulSoup(source_code, "lxml")
            rprts = html.findAll("table", {"class":"type5"})

            for items in rprts:
                
                titles = items.select(".title")
                for title in titles: 
                    title = title.get_text() 
                    title = re.sub('\n','',title)
                    title_result.append(title)

                article = items.select('.title')
                for li in article:
                    lis =  'https://finance.naver.com' + li.find('a')['href']
                    articles_code = requests.get(lis).text
                    htmls = BeautifulSoup(articles_code,"lxml")
                    #docs = htmls.find("table",{"class":"view"})
                    docs = htmls.find("div",{"class":"scr01"})
                    docs = docs.text.replace('/','').replace('?','').replace("\t",'').replace("\n",'').replace('/n','').replace('[','').replace(']','').replace('!','').replace('-','').replace('$','').replace('▲','').replace("'",'').replace('■','').replace('◆','').replace('#','').replace('_','').replace('=','').replace('"','').replace(" \'",'').replace('아웃링크','').replace('◀','').replace('▶','').replace('<','').replace('>','').replace(':','').replace(',','').replace('ⓒ','').replace('※','').replace('\xa0','').replace('&','').replace('△','').replace('이데일리','').replace('매일경제','').replace('파이낸셜뉴스','').replace('서울경제','').replace('한국경제','').replace('조선비즈','').replace('아시아경제','').replace('머니투데이','').replace('헤럴드경제','').replace('···','').replace('·','').replace('‘','').replace('’','').replace('..','').replace("“",'').replace("”",'').replace('`','').replace('…','').replace('Copyrights','').replace('━','').replace('@','').lstrip()
                    article_result.append(docs)
                #print(article_result)

                links = items.select('.title') 
                for link in links: 
                    add = 'https://finance.naver.com' + link.find('a')['href']
                    link_result.append(add)
                #print(link_result)

                dates = items.select('.date') 
                for date in dates:
                    date = date.get_text()
                    date_result.append(date)
                #print(date_result)


            result= {"date" : date_result, "headline" : title_result, "content" : article_result, "url" : link_result,"ticker":plusUrl.zfill(6)} 
            df_result = pd.DataFrame(result)
            time.sleep( random.uniform(2,4) )



            #df_result['date']=pd.to_datetime(df_result['date'].astype(str), format='%Y/%m/%d')
            #df_result.set_index('date', inplace=True)
            #print(df_result['date'])
        return df_result
                        



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class NewsDto(db.Model):
    __tablename__ = 'korea_recent_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    content : str = db.Column(db.Text)
    url :str = db.Column(db.String(500))
    ticker : str = db.Column(db.String(30))
    
    def __init__(self, id, date, headline, content, url, ticker):
        self.id = id
        self.date = date
        self.headline = headline
        self.content = content
        self.url = url
        self.ticker = ticker
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            content={self.content},url={self.url},ticker={self.ticker}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'content':self.ccontent,
            'url':self.url,
            'ticker':self.ticker
        }

class NewsVo:
    id : int = 0
    date: str =''
    headline: str=''
    content: str=''
    url: str =''
    ticker: str =''



Session = openSession()
session= Session()




class RNewsDao(NewsDto):
    
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    

    def bulk(self):
        path = self.data
        #kn = KoreaNews()
        #kn.new_model()
        companys = ['lg화학','lg이노텍']
        for com in companys:
            print(f'company:{com}')
            #df = kn.search_news(com)
            if com =='lg화학':
                com ='lgchem'
            elif com =='lg이노텍':
                com='lginnotek'
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            #df.to_csv(path + '/'+com+'_recent_news.csv',encoding='UTF-8')
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            print(df.head()) 
            session.bulk_insert_mappings(NewsDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(NewsDto.id)).one()

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
        data = cls.qeury.get(id)
        db.session.delete(data)
        db.session.commit()
    
    @classmethod
    def find_all(cls):
        sql =cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_all_by_ticker(cls, rnews):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']== rnews.ticker]
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
    def find_by_ticker(cls,ticker):
        return session.query(NewsDto).filter(NewsDto.ticker.like(ticker)).all()
    
    @classmethod
    def login(cls,news):
        sql = cls.query.fillter(cls.id.like(news.id)).fillter(cls.headline.like(news.headline))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('============================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))


# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('headline', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('content', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('url', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=True, help='This field cannot be left blank')



class RNews(Resource):

    @staticmethod
    def post(self):
        data = self.parser.parse_args()
        rnews = NewsDto(data['date'],data['headline'],data['content'],data['url'],data['ticker'])
        try:
            rnews.save(data)
            return {'code':0, 'message':'SUCCESS'},200
        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return rnews.json(), 201
    

    def get(self, ticker):
        rnews = RNewsDao.find_by_ticker(ticker)
        if rnews:
            return rnews.json()
        return {'message': 'The recent news was not found'}, 404


    def put(self, id):
        data = RNews.parser.parse_args()
        rnews = RNewsDao.find_by_id(id)

        rnews.date = data['date']
        rnews.ticker = data['ticker']
        rnews.url = data['url']
        rnews.headline = data['headline']
        rnews.content = data['content']
        rnews.save()
        return rnews.json()


class RNews_(Resource):
    def get(self):
        return RNewsDao.find_all(),200
    
    # @staticmethod
    # def post():
    #     rn = RNewsDao()
    #     rn.insert('korea_recent_news')
    
    # @staticmethod
    # def get():
    #     data = RNewsDao.find_all()
    #     return data, 200


class lgchemNews(Resource):
    
    @staticmethod
    def get():
        print("lgchem_recent_news get")
        rnews= NewsVo()
        rnews.ticker ='051910'
        data = RNewsDao.find_all_by_ticker(rnews)
        return data, 200

    @staticmethod
    def post():
        print('lgchem_recent_news post')
        args = parser.parse_args()
        rnews = NewsVo()
        rnews.ticker = args.ticker
        data = RNewsDao.find_all_by_ticker(rnews)
        return data[0],200


class lginnoteknews(Resource):
    
    @staticmethod
    def get():
        print("lginnotek_recent_news get")
        rnews= NewsVo()
        rnews.ticker = '011070'
        data = RNewsDao.find_all_by_ticker(rnews)
        return data, 200

    @staticmethod
    def post():
        print("lginnotek_recent_news post")
        args = parser.parse_args()
        rnews = NewsVo()
        rnews.ticker = args.ticker
        data = RNewsDao.find_all_by_ticker(rnews)
        return data[0], 200



