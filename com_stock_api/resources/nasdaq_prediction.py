from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.uscovid import USCovidDto, USCovids, USNewCases, CANewCases
from com_stock_api.resources.investingnews import InvestingDto, AppleSentiment, TeslaSentiment
from com_stock_api.resources.yhfinance import YHFinanceDto, AppleGraph, TeslaGraph
from sqlalchemy import and_,or_,func, extract
import os
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
import json
from datetime import datetime
from sklearn import preprocessing
import numpy as np
from com_stock_api.utils.file_helper import FileReader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


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

    @classmethod
    def delete(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        
    @classmethod
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
# =================      Deep Learning    =====================
# =============================================================
# =============================================================

#PRE: Create DB, Insert All preprocessed data into DB
class NasdaqDF():

    path : str 
    tickers : str
    # cols : ticker,date,open,high,low,close,adjclose,volume
    def __init__(self):
        self.path = os.path.abspath(__file__+"/.."+"/saved_data")
        self.fileReader = FileReader()
        self.df = None
        self.ticker = ''
        self.tickers= ['AAPL', 'TSLA']

    def hook(self):
        
        for tic in self.tickers:
            self.ticker = tic
            df = self.create_dataframe()
            self.train(df)
            # self.scatter_graph()
            # self.heatmap_graph()
            # self.draw_initial_graph(df)

    #Step1. Collect all the features which would affect on prediction to the one CSV file 
    def create_dataframe(self):
        main_df = self.df
        
        #1. Bring history of a chosen ticker
        df=pd.read_sql_table('Yahoo_Finance', engine.connect())
        df = df.loc[(df['ticker'] == self.ticker) & (df['date'] > '2019-12-31')& (df['date'] < '2020-07-01')]
        df = df.drop(['ticker', 'id'], axis=1)

        #1-1. Adding three more features: moving average, increase in volume(%), increase in adj_close(%)
        avg = []
        df['moving_avg'] = df['adjclose'].rolling(window=50, min_periods=0).mean()
        
        i =1
        vol_rate = [0]
        adjclose_rate = [0]

        while i < len(df):
            vol_rate.append((df.iloc[i]['volume']- df.iloc[i-1]['volume'])/df.iloc[i-1]['volume'])
            adjclose_rate.append((df.iloc[i]['adjclose']-df.iloc[i-1]['adjclose'])/df.iloc[i-1]['adjclose'])
            i+=1
        
        df['increase_rate_vol'] = vol_rate
        df['increase_rate_adjclose'] =  adjclose_rate 
       
        #1-2. Bring history of the other ticker
        df4=pd.read_sql_table('Yahoo_Finance', engine.connect())
        tic = [t for t in self.tickers if t != self.ticker]
        df4 = df4.loc[(df4['ticker'] == tic[0]) & (df4['date'] > '2019-12-31')& (df4['date'] < '2020-07-01')]
        df4 = df4.rename(columns={'open': tic[0]+'_open', 'high':tic[0]+'_high', 'low':tic[0]+'_low','close': tic[0]+'_close', 'adjclose': tic[0]+'_adjclose'})
        df4 = df4.drop(['ticker', 'id', 'volume'], axis=1)


        temp_df = pd.DataFrame()
        df5 = pd.read_sql_table('korea_finance', engine.connect())
        k_tickers = {'lg_chem': '051910', 'lg_innotek':'011070'} #LG Chem & LG Innotek 
        for k_tic, v in k_tickers.items():
            df5 = pd.read_sql_table('korea_finance', engine.connect())
            
            df5['date']= pd.to_datetime(df5['date'])
            df5 = df5.loc[(df5['ticker'] == v) & (df5['date'] > '2019-12-31')& (df5['date'] < '2020-07-01')]
            df5 = df5.rename(columns={'open': k_tic+'_open', 'close':  k_tic+'_close', 'high': k_tic+'_high', 'low': k_tic+'_low'})
            df5 = df5.drop(['id','ticker','volume'], axis=1)
            df5 = df5[df5['date'].notnull() == True].set_index('date')

            temp_df = temp_df.join(df5, how='outer')
  
        temp_df['date'] = temp_df.index


        #2. Bring news sentiment 
        if (self.ticker == 'AAPL'):
            apple_json = json.dumps(AppleSentiment.get()[0], default = lambda x: x.__dict__)
            df2 = pd.read_json(apple_json)
            df2 = df2.drop(['id'], axis=1)

        else:
            tesla_json = json.dumps(TeslaSentiment.get()[0], default = lambda x: x.__dict__)
            df2 = pd.read_json(tesla_json)
            df2 = df2.drop(['id'], axis=1)

        #3. Bring US_covid status per day
        covid_json = json.dumps(USCovids.get()[0], default = lambda x: x.__dict__)
        df3 = pd.read_json(covid_json)
        df3 = df3.loc[(df3['date'] > '2019-12-31') & (df3['date'] < '2020-07-01')]
        df3 = df3.drop(['id', 'total_cases', 'total_deaths', 'ca_cases', 'ca_deaths'], axis=1)

        #4. Combine all features in one csv file
        df = df[df['date'].notnull() == True].set_index('date')
        df4 = df4[df4['date'].notnull() == True].set_index('date')
        temp_df = temp_df[temp_df['date'].notnull()==True].set_index('date')

        df2 = df2[df2['date'].notnull() == True].set_index('date')
        df3 = df3[df3['date'].notnull() == True].set_index('date')

        main_df = df.join(df4, how='outer')
        # main_df = main_df.join(temp_df, how='outer')
        main_df = main_df.join(df2, how='outer')
        main_df = main_df.join(df3, how='outer')
        main_df[['new_us_cases', 'new_us_death', 'new_ca_cases', 'new_ca_death']] = main_df[['new_us_cases', 'new_us_death', 'new_ca_cases', 'new_ca_death']].fillna(value=0)

        #6. Save to CSV file
        output_file = self.ticker + '_dataset.csv'
        result = os.path.join(self.path, output_file)
        main_df.to_csv(result)
        return main_df

        '''
  Index(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'moving_avg',
       'increase_rate_vol', 'increase_rate_adjclose', 'AAPL_open', 'AAPL_high',
       'AAPL_low', 'AAPL_close', 'AAPL_adjclose', 'lg_chem_open',
       'lg_chem_close', 'lg_chem_high', 'lg_chem_low', 'lg_innotek_open',
       'lg_innotek_close', 'lg_innotek_high', 'lg_innotek_low', 'neg', 'pos',
       'neu', 'compound', 'new_us_cases', 'new_us_death', 'new_ca_cases',
       'new_ca_death'],

        '''

    def draw_initial_graph(self, df):
        
        # df=pd.read_sql_table('Yahoo_Finance', engine.connect(), parse_dates=['date'])
        # df = df.loc[(df['ticker'] == self.ticker) & (df['date'] > '2009-12-31')& (df['date'] < '2020-07-01')]
        
        cols = ['open', 'low', 'close', 'adjclose' , 'moving_avg', 'increase_rate_vol', 'increase_rate_adjclose']
        path = os.path.abspath(__file__+"/.."+"/plots/")
        
        # Getting only business days
        isBusinessDay = BDay().onOffset
        df['date'] = df.index
        match_series = pd.to_datetime(df['date']).map(isBusinessDay)
        df = df[match_series]
        print(df)        
        for c in cols:
            title = self.ticker + " " + c + " graph"
            ax = df.plot(x='date', y=c, colormap='Set2', title = title)
            if 'rate' in c:
                ax.set(xlabel="Date", ylabel="percentage (%)")
                ax.hlines(y=0, xmin='2020-01-01', xmax='2020-07-01', colors='r', linestyles='--', lw=2)
            else:
                ax.set(xlabel="Date", ylabel="$ U.S. Dollar")
            # plt.show()
            file_name = title + ".png"
            output_file = os.path.join(path, file_name)
            plt.savefig(output_file)

    def scatter_graph(self):
        path = os.path.abspath(__file__+"/.."+"/plots/")        
        filen = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, filen)
       
        df = pd.read_csv(input_file)
        df.drop(['date'], axis=1)
        tic = [t for t in self.tickers if t !=self.ticker]
        op_tic = tic[0]

        sns.pairplot(df, height = 2.5)
        plt.tight_layout()
        plt.title("The Scatter Plot of "+ self.ticker)
        file_name = self.ticker + "_correlation.png"
        output_file = os.path.join(path, file_name)
        plt.savefig(output_file)
        print('=== Saved scatter plot ===')
        

    def heatmap_graph(self):
        path = os.path.abspath(__file__+"/.."+"/plots/")

        filen = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, filen)
        df = pd.read_csv(input_file, header=0)
        df.drop(['date'], axis=1, inplace=True)
        print(df.columns)
        tic = [t for t in self.tickers if t !=self.ticker]
        op_tic = tic[0]

        sns.heatmap(df)
        plt.title('Heatmap of ' + self.ticker, fontsize=20)
        
        file_name2 = self.ticker + "_heatmap.png"
        output_file2 = os.path.join(path, file_name2)
        plt.savefig(output_file2)
        print('=== Saved heatmap ===')
        
    def train(self, df):
        df= df.dropna()
        X = df.drop('adjclose', axis=1)
        y= df.adjclose
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
        print(X_train)
        print(len(X_train), "train + ", len(X_test), "test")        

if __name__ == "__main__":
    here = NasdaqDF()
    here.hook()

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