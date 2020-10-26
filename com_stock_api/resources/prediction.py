from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.resources.uscovid import USCovidDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from com_stock_api.resources.investingnews import InvestingDto
import os
import pandas as pd

class PredictionDto(db.Model):
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
        return f'Prediction(id=\'{self.id}\',ticker=\'{self.ticker}\',date=\'{self.date}\',\
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


class PredictionDao(PredictionDto):

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_date(cls, date):
        return cls.query.filer_by(date == date).all()

    @staticmethod   
    def insert_many():
        Session = openSession()
        session = Session()
        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/data/")
            file_name = tic + '_pred.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(PredictionDto, df.to_dict(orient="records"))
            session.commit()
        session.close()


# =============================================================
# =============================================================
# ======================      CONTROLLER    ======================
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

class Prediction(Resource):    
    @staticmethod
    def post():
        data = parser.parse_args()
        prediction = PredictionDto(data['date'], data['ticker'],data['pred_price'], data['stock_id'], data['covid_id'], data['news_id'])
        try: 
            prediction.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200
        except:
            return {'message': 'An error occured inserting the article'}, 500
        return prediction.json(), 201
    
    
    def get(self, id):
        article = PredictionDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404

    def put(self, id):
        data = Prediction.parser.parse_args()
        prediction = PredictionDao.find_by_id(id)

        prediction.title = data['title']
        prediction.content = data['content']
        prediction.save()
        return prediction.json()

class Predictions(Resource):
    def get(self):
        return {'predictions': list(map(lambda article: article.json(), PredictionDao.find_all()))}
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}