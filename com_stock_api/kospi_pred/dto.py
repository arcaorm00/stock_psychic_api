from com_stock_api.ext.db import db 
from com_stock_api.korea_covid.dto import KoreaDto
from com_stock_api.naver_finance.dto import StockDto
from com_stock_api.naver_news.dto import NewsDto

class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    kospi_id : int = db.Column(db.VARCHAR(30), primary_key = True, index=True)
    date : int = db.Column(db.DATETIME)
    stock :int = db.Column(db.VARCHAR(30))
    price : int = db.Column(db.VARCHAR(30))

    date: int = db.Column(db.DATETIME, db.ForeignKey(KoreaDto.date))
    date: int = db.Column(db.DATETIME, db.ForeignKey(StockDto.date))
    date: int = db.Column(db.DATETIME, db.ForeignKey(NewsDto.date))


    def __init__(self, kospi_id, date, stock, price):
        self.kospi_id = kospi_id
        self.date = date
        self.stock = stock
        self.price = price
    
    def __repr__(self):
        return f'kospi_id={self.kospi_id}, date={self.date}, stock={self.stock},\
            price={self.price}'
            
    @property
    def json(self):
        return {
            'kospi_id': self.kospi_id,
            'date': self.date,
            'stock' : self.stock,
            'price' : self.price
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()




  