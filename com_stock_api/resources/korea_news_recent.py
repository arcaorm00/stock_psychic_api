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


class KoreaNews():
    
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

    def search_news(self,company):
        print(f'ENTER STEP 2 : search_news ')
        print(f'company : {company}')

        article_result =[]
        title_result = []
        link_result = []
        date_result = []
        #source_result = []

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
            
                    #docs = docs.get_text()
                

                    #print(type(docs))
                    #print(docs)
                    article_result.append(docs)
                #print(article_result)

                links = items.select('.title') 
                for link in links: 
                    add = 'https://finance.naver.com' + link.find('a')['href']
                    link_result.append(add)
                #print(link_result)

                dates = items.select('.date') 
                #date_result = [date.get_text() for date in dates] 
                for date in dates:
                    date = date.get_text()
                    date_result.append(date)
                #print(date_result)

                # sources = items.select('.info')
                # #source_result = [source.get_text() for source in sources]
                # for source in sources:
                #     source = source.get_text()
                #     source_result.append(source)
                # #print(source_result)


            result= {"date" : date_result, "headline" : title_result, "content" : article_result, "url" : link_result,"ticker":plusUrl.zfill(6)} 
            # press" : source_result
            df_result = pd.DataFrame(result)
            #df_result['date']=pd.to_datetime(df_result['date'].astype(str), format='%Y/%m/%d')
            #df_result.set_index('date', inplace=True)
            #print(df_result['date'])
        return df_result
                        




class NewsDto(db.Model):
    __tablename__ = 'korea_recent_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    content : str = db.Column(db.Text) #String(10000)
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




class RecentNewsDao(NewsDto):
    
    # def __init__(self):
    #     self.data = os.path.abspath(__file__+"/.."+"/data/")
    
    @staticmethod
    def bulk(): #self
        kn = KoreaNews()
        kn.new_model()
        companys = ['lg화학','lg이노텍']
        for com in companys:
            df = kn.search_news(com)
            #df = service.hook()
            # path = self.data
            # df=pd.read_csv( path +'/011070.csv',encoding='utf-8',dtype=str)
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
    RecentNewsDao.bulk()
    #n = RecentNewsDao()
    #n.bulk()


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
            news = RecentNewsDao.find_by_id(id)
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
        rn = RecentNewsDao()
        rn.insert('naver_news')
    
    @staticmethod
    def get():
        data = RecentNewsDao.find_all()
        return data, 200

class Auth(Resource):
    @staticmethod
    def post():
        body = request.get_json()
        news = NewsDto(**body)
        RecentNewsDao.save(news)
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
        data = RecentNewsDao.login(news)
        return data[0], 200