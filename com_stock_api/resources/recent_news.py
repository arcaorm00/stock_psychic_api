from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import os
import re
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import urllib
from urllib.request import Request, urlopen
import requests
from datetime import datetime, timedelta
from http.client import IncompleteRead
from unicodedata import normalize
from newspaper import Article
from newspaper.article import ArticleException
from newspaper import Config
from sqlalchemy import and_,or_,func
import json

# =============================================================
# =============================================================
# ======================      SERVICE    ======================
# =============================================================
# =============================================================

class RecentNewsPro:
    tickers: str = ['AAPL', 'TSLA']
    ticker : str
    finviz_url = 'https://finviz.com/quote.ashx?t='

    def __init__(self):
        ...
    
    def hook(self):
        dfs=[]
        for t in self.tickers:
            self.ticker = t
            lst = self.get_latest_news()
            self.save_news(lst)
            df= self.df_processing(lst)
            dfs.append(df)
        return dfs
        print( " ++ DONE ++ ")

    def get_latest_news(self):
        ticker = self.ticker
        url = self.finviz_url

    # Open url
        url +=ticker
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'

        req = Request(url=url, headers={'user-agent': user_agent})
        resp = urlopen(req)
        html = bs(resp, features="lxml")
        news_table = html.find(id='news-table')

    # Extract news data only

        df = news_table.findAll('tr')
        today = datetime.now()
        recent = today - timedelta(days = 3)
        processed_data = []
        
        for i, row in enumerate(df):
            #Basic given information from finviz.com
            headline = row.a.text
            time = row.td.text
            time = time.strip()
            link = row.find("a").get("href")
            
            # Get published date and time
            date_time = self.get_published_datetime(time)
            published_date = date_time[0] if (date_time[0]!=0) else (processed_data[-1][0])
            published_time = date_time[1]
           
            #Get news content by using links from finviz
            if "https://finance.yahoo.com/news" in link:
                print("Getting yahoo news...", i,"/", len(df))
                page = self.get_yahoo_page(link)
                content=self.get_yahoo_news(page)
                image = self.get_yahoo_image(page)
            else:
                config = Config()
                config.browser_user_agent = user_agent
                article = Article(link, config=config)
                try:
                    article.download() 
                except ArticleException as ae:
                    print (ae)
                    continue
                except Exception as e:
                    print(e)
                    continue

                else:
                    print("Getting other news...", i,"/", len(df))
                    article.parse()
                    content = self.clean_paragraph(article.text)
                    content = "".join(content)
                    image = article.top_image
          

            #Collect news data in a list
            processed_data.append([published_date, published_time, self.ticker, link, headline, image, content[:200]+'...'])
            
            if (datetime.strptime(published_date, '%Y-%m-%d') < recent):
                processed_data.pop()
                break
        return processed_data

    def get_published_datetime(self, time):
        
        try:
            publish_date = str(datetime.strptime(time, '%b-%d-%y %I:%M%p'))
            publish_date = "".join(publish_date)
            pub_time = publish_date[11:]
            publish_date = publish_date[:10]

        except ValueError:
            publish_date = 0
            pub_time = str(datetime.strptime(time, '%I:%M%p'))
            pub_time = "".join(pub_time)[11:]

        return publish_date, pub_time

    def get_yahoo_page(self, link):
        request = Request(link, headers={"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
        content = urlopen(request).read()
        return bs(content, 'lxml')


    def get_yahoo_news(self, page):
        text_tag = page.find('div', attrs={'class': 'caas-body'})
        paragraphs = text_tag.find_all('p')
        text = '\n'.join([self.clean_paragraph(p.get_text()) for p in paragraphs[:-1]])
        text = "".join(text)
        return text

    def get_yahoo_image(self, page):
        image_tags = page.find_all(attrs={'caas-img'})
        return image_tags[-1].get('src')

    def clean_paragraph(self, paragraph):
        paragraph = re.sub(r'\(http\S+', '', paragraph)
        paragraph = re.sub(r'\([A-Z]+:[A-Z]+\)', '', paragraph)
        paragraph = re.sub(r'[\n\t\s\']', ' ', paragraph)
        return normalize('NFKD', paragraph)    

    def save_news(self, data):
        col = ['date', 'time', 'ticker', 'link', 'headline', 'image', 'content']
        df = pd.DataFrame(data, columns=col)
        path = os.path.abspath(__file__+"/.."+"/saved_data/")
        file_name = self.ticker + '_recent_news.csv'
        output_file = os.path.join(path,file_name)
        df.to_csv(output_file)
        print("Completed saving ", file_name)
    
    def df_processing(self, data):
        col = ['date', 'time', 'ticker', 'link', 'headline', 'image','content']
        df = pd.DataFrame(data, columns=col)
        return df

# if __name__=='__main__':
#     news_pro = RecentNewsPro()
#     news_pro.hook()

# =============================================================
# =============================================================
# ===================      Modeling    =======================
# =============================================================
# =============================================================

class RecentNewsDto(db.Model):
    __tablename__ = 'Recent_News'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date: str = db.Column(db.Date)
    time: str = db.Column(db.Time())
    ticker: str = db.Column(db.String(30))
    link: str = db.Column(db.Text)
    headline: str = db.Column(db.String(225))
    image: str = db.Column(db.Text)
    content : str = db.Column(db.Text)
    #date format : YYYY-MM-DD
    
    def __init__(self, date, time, ticker, link, headline, image, content):
        self.date = date
        self.time = time
        self.ticker = ticker
        self.link = link
        self.headline = headline
        self.image = image
        self.content = content

    def __repr__(self):
        return f'RecentNews(id=\'{self.id}\', date=\'{self.date}\', time=\'{self.time}\',\
            ticker=\'{self.ticker}\',link=\'{self.link}\', headline=\'{self.headline}\'\
                image=\'{self.image}\', content=\'{self.content}\')'


    @property
    def json(self):
        return {
            'id' : self.id,
            'date' : self.date,
            'time' : self.time,
            'ticker' : self.ticker,
            'link' : self.link,
            'headline' : self.headline,
            'image' : self.image,
            'content' : self.content
        }

class RecentNewsVo:
    id: int = 0
    date: str = ''
    time : str = ''
    ticker: str = ''
    link: str = ''
    headline: str = ''
    image: str = ''
    content: str = ''

Session = openSession()
session = Session()
class RecentNewsDao(RecentNewsDto):

    @staticmethod
    def count():
        return session.query(func.count(RecentNewsDto.id)).one()

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @staticmethod   
    def bulk():
        # service = RecentNewsPro()
        # dfs = service.hook()
        # for i in dfs:
        #     print(i.head())
        #     session.bulk_insert_mappings(RecentNewsDto, i.to_dict(orient="records"))
        #     session.commit()
        # session.close()

        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/saved_data/")
            file_name = tic + '_recent_news.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(RecentNewsDto, df.to_dict(orient="records"))
            session.commit()
        session.close()

    @staticmethod
    def save(news):
        db.session.add(news)
        db.session.commit()

    @staticmethod
    def delete(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()

    
    @classmethod
    def find_by_date(cls, date, tic):
        return session.query(RecentNewsDto).filter(and_(RecentNewsDto.date.like(date), RecentNewsDto.ticker.ilike(tic)))
    @classmethod
    def find_by_ticker(cls, tic):
        return session.query(RecentNewsDto).filter(RecentNewsDto.ticker.ilike(tic))
    @classmethod
    def find_by_period(cls,tic, start_date, end_date):
        return session.query(RecentNewsDto).filter(and_(RecentNewsDto.ticker.ilike(tic),date__range=(start_date, end_date)))
    @classmethod
    def find_today_one(cls, tic):
        today = datetime.today()
        return session.query(RecentNewsDto).filter(and_(RecentNewsDto.ticker.ilike(tic),RecentNewsDto.date.like(today)))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']==stock.ticker]
        return json.loads(df.to_json(orient='records'))

# =============================================================
# =============================================================
# ======================   RESOURCING    ======================
# =============================================================
# =============================================================
parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('time', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('link', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('image', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('content', type=str, required=False, help='This field cannot be left blank')

class RecentNews(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        recent_news = RecentNewsDto(data['date'], data['time'] ,data['ticker'], data['link'],data['headline'], data['image'], data['content'])
        try: 
            recent_news.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return recent_news.json(), 201
          
    @staticmethod
    def get(ticker):
        args = parser.parse_args()
        print("=====recent_news.py / recent_news' get")
        stock = RecentNewsVo
        stock.ticker = ticker
        data = RecentNewsDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def put():
        data = RecentNews.parser.parse_args()
        stock = RecentNewsDao.find_by_id(id)

        stock.date = data['date']
        stock.time = data['time']
        stock.ticker = data['ticker']
        stock.link = data['link']
        stock.headline = data['headline']
        stock.image = data['image']
        stock.content = data['content']
        stock.save()
        return stock.json()

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Headline {args["headline"]} on date {args["date"]} deleted')
        RecentNewsDao.delete(args['id'])
        return {'code' : 0 , 'message' : 'SUCCESS'}, 200

class AppleNews(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        recent_news = RecentNewsDto(data['date'], data['time'] ,data['ticker'], data['link'],data['headline'], data['image'], data['content'])
        try: 
            recent_news.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return recent_news.json(), 201
          
    @staticmethod
    def get():
        print("=====recent_news.py / tesla_news' get")
        stock = RecentNewsVo
        stock.ticker = 'AAPL'
        data = RecentNewsDao.find_all_by_ticker(stock)
        return data, 200
    @staticmethod
    def put(id):
        data = RecentNews.parser.parse_args()
        stock = RecentNewsDao.find_by_id(id)

        stock.date = data['date']
        stock.time = data['time']
        stock.ticker = data['ticker']
        stock.link = data['link']
        stock.headline = data['headline']
        stock.image = data['image']
        stock.content = data['content']
        stock.save()
        return stock.json()

class TeslaNews(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        recent_news = RecentNewsDto(data['date'], data['time'] ,data['ticker'], data['link'],data['headline'], data['image'], data['content'])
        try: 
            recent_news.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return recent_news.json(), 201
          
    @staticmethod
    def get():
        print("=====recent_news.py / tesla_news' get")
        stock = RecentNewsVo
        stock.ticker = 'TSLA'
        data = RecentNewsDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def put(id):
        data = RecentNews.parser.parse_args()
        stock = RecentNewsDao.find_by_id(id)

        stock.date = data['date']
        stock.time = data['time']
        stock.ticker = data['ticker']
        stock.link = data['link']
        stock.headline = data['headline']
        stock.image = data['image']
        stock.content = data['content']
        stock.save()
        return stock.json()
