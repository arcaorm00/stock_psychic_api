from com_stock_api.ext.db import db, openSession, engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import os
import re
from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import requests
from unicodedata import normalize
from datetime import datetime, date
from http.client import IncompleteRead
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import csv
from flask_restful import Resource, reqparse
from sqlalchemy import and_,or_,func
import json
from matplotlib import pyplot as plt


# =============================================================
# =============================================================
# ======================      SERVICE    ======================
# =============================================================
# =============================================================

class InvestingPro:
    tickers = {'AAPL' : 'apple-computer-inc', 'TSLA' : 'tesla-motors'}
    ticker_str : str
    ticker : str
    n : int
    # Temporaily set up the page number for the time sake
    AAPL = 240
    TSLA = 200
    START_DATE = datetime.strptime('2020-01-01', '%Y-%m-%d')
    END_DATE = datetime.strptime('2020-06-30', '%Y-%m-%d')
    def __init__(self, ticker):
        if (self.tickers[ticker]) != None:
            self.ticker_str = self.tickers.get(ticker)
            self.ticker = ticker
            self.n = self.AAPL if self.ticker == 'AAPL' else self.TSLA
            self.processed_info = []
        else:
            print(KeyError)
    def hook(self):
        news_pages  =[]
        while(self.n>1):
            news_page = self.get_stock_news_pages(self.ticker_str, self.n)
            # Get all the pages within the date range (Possible to include other dates \
            # in first and last news page due to multiple articles on one page)
            if (self.dates_checker(news_page)):
                print("Getting ",self.n,"th news page...")
                news_pages.append(news_page)
            else:
                #Device to take START_DATE pages and to stop bringing pages when it reaches to the END_DATE
                if (len(news_pages) > 0):
                    break
            self.n-=1
        article_pages=[]
        i = 1
        for index, p in enumerate(news_pages):
            links = self.get_internal_article_links(p)
            print('Processing ... {} / {}'.format(i,len(news_pages) ))
            for l in reversed(links):
                page = self.get_article_page(l)
                if (index ==0 or index == len(news_page) -1):
                    t= datetime.strptime(self.get_publish_time(page), '%Y-%m-%d')
                    if (t > self.END_DATE or t < self.START_DATE):
                        continue
                text = self.extract_text(page)
                headline = self.get_headline(page)
                date = self.get_publish_time(page)
                self.processed_info.append([self.ticker, date, l, headline, text])
            i+=1
        self.save_news(self.processed_info)
        self.get_sentiment_analysis(self.processed_info)
        print("++ DONE ++")
    def save_news(self, news_list):
        col = ['ticker', 'date', 'link', 'headline', 'content']
        df = pd.DataFrame(news_list, columns=col)
        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = self.ticker + '_news.csv'
        output_file = os.path.join(path,file_name)
        df.to_csv(output_file)
        print("Completed saving ", file_name)
    def get_stock_news_pages(self, stock_string, n):
        # pages =[]
        request = Request('https://www.investing.com/equities/' + stock_string + '-news/' + str(n), headers={"User-Agent": "Mozilla/5.0"})
        content = urlopen(request).read()
        # pages.append(bs(content, 'html.parser'))
        return bs(content, 'html.parser')
    def dates_checker (self, page):
        tags = page.find_all('span', attrs={'class' : 'date'})[:10]
        dates = [i.text.strip().replace("-", '').replace('\xa0','') for i in tags]
        published_dates=[]
        for i in dates:
            try:
                published_dates.append(datetime.strptime(i, '%b %d, %Y'))
            except ValueError:
                pass
        published_dates[:] = [date for date in published_dates if (self.START_DATE <= date <= self.END_DATE)]
        return False if (len(published_dates) == 0) else True    
    def get_headline(self, page):
        headline = page.find('title').text
        return headline
    def get_internal_article_links(self, page):
        news = page.find_all('div', attrs={'class': 'mediumTitle1'})[1]
        articles = news.find_all('article', attrs={'class': 'js-article-item articleItem'})
        return ['https://www.investing.com' + a.find('a').attrs['href'] for a in articles]
    def get_article_page(self, article_link):
        request = Request(article_link, headers={"User-Agent": "Mozilla/5.0"})
        content = urlopen(request).read()
        try:
            page= bs(content, 'lxml')
        except IncompleteRead as e:
            print("Incomplete Read error occurs")
            page = e.partial
        return page
    def clean_paragraph(self, paragraph):
        paragraph = re.sub(r'\(http\S+', '', paragraph)
        paragraph = re.sub(r'\([A-Z]+:[A-Z]+\)', '', paragraph)
        paragraph = re.sub(r'[\n\t\s\']', ' ', paragraph)
        return normalize('NFKD', paragraph)    
    def extract_text(self, article_page):
        text_tag = article_page.find('div', attrs={'class': 'WYSIWYG articlePage'})
        paragraphs = text_tag.find_all('p')
        text = '\n'.join([self.clean_paragraph(p.get_text()) for p in paragraphs[:-1]])
        text = "".join(text)
        return text
    def get_publish_time(self, article_page):
        details = article_page.find('meta', attrs={'itemprop': 'dateModified'})
        published_date = details.get_attribute_list('conclasstent')[0]
        published_date = str(datetime.strptime(published_date, '%Y-%m-%d %H:%M:%S'))
        published_date = "".join(published_date)
        published_date = published_date[:10]
        return published_date
    def get_sentiment_analysis(self, news_list):
        vader = SentimentIntensityAnalyzer()
        col = ['ticker', 'date', 'link', 'headline', 'content']
        news_with_scores = pd.DataFrame(news_list, columns=col)
        # news_with_scores = pd.read_csv(news_list)
        # iterating the columns 
        # print(news_with_scores['content'].dtype)
        # processed = news_with_scores['content']
        scores =news_with_scores['content'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        news_with_scores = news_with_scores.join(scores_df, rsuffix='_right')
        news_with_scores['date'] = pd.to_datetime(news_with_scores.date).dt.date
        news_with_scores.drop('content',axis=1,inplace=True)
        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = self.ticker + '_sentiment.csv'
        output_file = os.path.join(path,file_name)
        news_with_scores.to_csv(output_file)
        print("Completed saving ", file_name)
        print(news_with_scores.head())
    
'''        
if __name__=='__main__':
    path = os.path.abspath(__file__+"/.."+"/data/")
    file_name = 'TSLA_news.csv'
    input_file = os.path.join(path,file_name)
    tesla = InvestingPro('TSLA')
    # tesla.get_sentiment_analysis(input_file)
    # apple = InvestingPro('AAPL')
    tesla.hook()
    # apple.hook()
    # tesla.get_graph(input_file)
'''

# =============================================================
# =============================================================
# ======================    MODELING    =======================
# =============================================================
# =============================================================

class InvestingDto(db.Model):
    __tablename__ = 'Investing_News'
    __table_args__={'mysql_collate':'utf8_general_ci'}
        # , primary_key = True, index = True

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.Date)
    ticker : str = db.Column(db.String(30)) #stock symbol
    link : str = db.Column(db.String(225))
    headline : str = db.Column(db.String(255))
    neg : float = db.Column(db.Float)
    pos : float = db.Column(db.Float)
    neu : float = db.Column(db.Float)
    compound :float  = db.Column(db.Float)

    def __init__(self, date, ticker, link, headline, neg, pos, neu, compound):
        self.date = date
        self.ticker = ticker
        self.link = link
        self.headline = headline
        self.neg = neg
        self.pos = pos
        self.neu = neu
        self.compound = compound

    def __repr__(self):
        return f'Investing(id=\'{self.id}\', date=\'{self.date}\',ticker=\'{self.ticker}\',\
                link=\'{self.link}\', headline=\'{self.headline}\',neg=\'{self.neg}\', \
                pos=\'{self.pos}\',neu=\'{self.neu}\', compound=\'{self.compound}\',)'


    
    def json(self):
        return {
            'id': self.id,
            'date' : self.date,
            'ticker' : self.ticker,
            'link' : self.link,
            'headline' : self.headline,
            'neg' : self.neg,
            'pos' : self.pos,
            'neu' : self.neu,
            'compound' : self.compound
        }

class InvestingVo:
    id: int = 0
    date: str = ''
    ticker: str = ''
    link: str = ''
    headline: str = ''
    neg: float = 0.0
    pos: float = 0.0
    neu: float = 0.0
    compound: float = 0.0

Session = openSession()
session = Session()

class InvestingDao(InvestingDto):

    @staticmethod
    def count():
        return session.query(func.count(InvestingDto.id)).one()

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_date(cls, date):
        return cls.query.filer_by(date == date).all()

    @staticmethod   
    def bulk():
        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/data/")
            file_name = tic + '_sentiment.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(InvestingDto, df.to_dict(orient="records"))
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
        return session.query(InvestingDto).filter(and_(InvestingDto.date.like(date), InvestingDto.ticker.ilike(tic)))
    @classmethod
    def find_by_ticker(cls, tic):
        return session.query(InvestingDto).filter(InvestingDto.ticker.ilike(tic))
        
    @classmethod
    def find_by_period(cls,tic, start_date, end_date):
        return session.query(InvestingDto).filter(and_(InvestingDto.ticker.ilike(tic) ,date__range=(start_date, end_date)))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']==stock.ticker]
        return json.loads(df.to_json(orient='records'))


# =============================================================
# =============================================================
# ======================     CONTROLLER    ====================
# =============================================================
# =============================================================
parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('link', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('neg', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('pos', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('neu', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('compound', type=float, required=False, help='This field cannot be left blank')

class Investing(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        news_sentiment = InvestingDto(data['date'], data['ticker'], data['link'],data['headline'], data['neg'], data['pos'], data['neu'], data['compound'])
        try: 
            news_sentiment.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting the news sentiment'}, 500
        return news_sentiment.json(), 201     
    
    @staticmethod
    def get(ticker):
        print("=====investing.py / Investing's get")
        args = parser.parse_args()
        stock = InvestingVo()
        stock.ticker = ticker
        data = InvestingDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def put(id):
        data = Investing.parser.parse_args()
        stock = InvestingDao.find_by_id(id)

        stock.date = data['date']
        stock.ticker = data['ticker']
        stock.link = data['link']
        stock.headline = data['headline']
        stock.neg = data['neg']
        stock.pos = data['pos']
        stock.neu = data['neu']
        stock.compound = data['compound']

        stock.save()
        return stock.json()

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Headline {args["headline"]} on date {args["date"]} deleted')
        InvestingDao.delete(args['id'])
        return {'code' : 0 , 'message' : 'SUCCESS'}, 200

#This class should operate after saving data into mariadb
class AppleSentiment(Resource):

    @staticmethod
    def get():
        print("=====investingnews.py / AppleSentiment's get")
        query = InvestingDao.find_by_ticker('AAPL')
        df = pd.read_sql_query(query.statement, query.session.bind, parse_dates=['date'])
        #Calculate mean values for each components; pos, neu, neg, compound
        means = df.resample('D', on='date').mean().dropna()
        means.insert(0, 'date', means.index)
        data = json.loads(means.to_json(orient='records'))
        return data,200

class TeslaSentiment(Resource):

    @staticmethod
    def get():
        print("=====investingnews.py / TeslaSentiment's get")
        query = InvestingDao.find_by_ticker('TSLA')
        df = pd.read_sql_query(query.statement, query.session.bind, parse_dates=['date'])
        #Calculate mean values for each components; pos, neu, neg, compound
        means = df.resample('D', on='date').mean().dropna()
        means.insert(0, 'date', means.index)
        data = json.loads(means.to_json(orient='records'))
        return data,200




class InvestingGraph(Resource):
    
    @staticmethod
    def draw_graph(ticker):

        tickers = {'AAPL':'Apple Inc.', 'TSLA':'Tesla Inc.' }
        if ticker == 'AAPL':
            result = json.dumps(AppleSentiment.get()[0], default = lambda x: x.__dict__)
        else:                
            result = json.dumps(TeslaSentiment.get()[0], default = lambda x: x.__dict__)

        df = pd.read_json(result)
        df =df.drop(['id'], axis=1)
        title = tickers[ticker]+" news sentiment for the first half of 2020"
        ax=df.plot(x='date', y='compound', figsize=(20,10))
        ax.set(xlabel="Date", ylabel="Overall scores", title=title)
        ax.hlines(y=0, xmin='2020-01-01', xmax='2020-07-01', colors='r', linestyles='--', lw=2)

        path = os.path.abspath(__file__+"/.."+"/plots/")
        file_name = ticker + "_news_sentiment_anaylsis.png"
        output_file = os.path.join(path, file_name)
        plt.savefig(output_file)
        
        plt.show()

if __name__=='__main__':
    apple = InvestingGraph()
    tickers = ['AAPL', 'TSLA']
    for t in tickers:
        apple.draw_graph(t)
#     result = apple.apple_dataframe()
#     apple.draw_graph(result)

