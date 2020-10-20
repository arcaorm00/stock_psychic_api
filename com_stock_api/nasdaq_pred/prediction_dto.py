from com_stock_api.ext.db import db
from com_stock_api.us_covid.us_covid_dto import USCovidDto
from com_stock_api.yhfinance.yhfinance_dto import YHFinanceDto
from com_stock_api.yhnews.yhnews_dto import YHNewsDto


class PredictionDto(db.Model):
    __tablename__ = 'NASDAQ_prediction'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date: str = db.Column(db.Date)
    ticker: str = db.Column(db.String(30))
    pred_price: float = db.Column(db.Float)
    
    stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    covid_id : int = db.Column(db.Integer, db.ForeignKey(USCovidDto.id))
    news_id: int = db.Column(db.Integer, db.ForeignKey(YHNewsDto.id))



    def __init__(self, ticker, date, pred_price, stock_id, covid_id, news_id):
        self.date = date
        self.ticker = ticker
        self.pred_price = pred_price

        self.stock_id = stock_id
        self.covid_id = covid_id
        self.news_id = news_id

    def __repr__(self):
        return f'Prediction(id=\'{self.id}\',date=\'{self.date}\',\
            ticker=\'{self.ticker}\',date=\'{self.date}\',pred_price=\'{self.pred_price}\',\
                stock_id=\'{self.stock_id}\',covid_id=\'{self.covid_id}\', news_id=\'{self.news_id}\' )'

    @property
    def json(self):
        return {
            'id' : self.id,
            'date' : self.date,
            'ticker' : self.ticker,
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