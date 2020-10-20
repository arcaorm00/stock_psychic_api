from com_stock_api.ext.db import db 
from com_stock_api.korea_covid.dto import KoreaDto
from com_stock_api.naver_finance.dto import StockDto
from com_stock_api.naver_news.dto import NewsDto

class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : int = db.Column(db.DATETIME)
    stock :int = db.Column(db.VARCHAR(30))
    price : int = db.Column(db.VARCHAR(30))

    covid_id: int = db.Column(db.Integer, db.ForeignKey(KoreaDto.id))
    stock_id: int = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    news_id: int = db.Column(db.Integer, db.ForeignKey(NewsDto.id))


    def __init__(self,id,date, covid_id,stock_id,news_id,stock, price):
        self.id = id
        self.date = date
        self.covid_id = covid_id
        self.stock_id= stock_id
        self.news_id= news_id
        self.stock = stock
        self.price = price
    
    def __repr__(self):
        return f'id={self.id},date={self.date},covid_id ={self.covid_id },stock_id={self.sstock_id},news_id={self.news_id},  stock={self.stock},\
            price={self.price}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'covid_id': self.covid_id,
            'stock_id': self.stock_id,
            'news_id': self.news_id,
            'stock' : self.stock,
            'price' : self.price
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()




  