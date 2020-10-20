from com_stock_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import VARCHAR

class Stock(Base):
    
    __tablename__ = 'naver_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    stock_id =Column(String(30),primary_key = True, index = True)
    date = Column(DATE)
    open = Column(VARCHAR(30))
    close = Column(VARCHAR(30))
    high = Column(VARCHAR(30))
    low = Column(VARCHAR(30))
    amount = Column(VARCHAR(30))
    stock = Column(VARCHAR(30))

    def __repr__(self):
        return 'Stock(stock_id={},date={},open={},close={},high={},low={},amount={},stock={}'\
            .format(self.id,self.data,self.open,self.close,self.high,self.low,self.amount,self.stock)

    @property
    def serialize(self):
        return {
            'stock_id':self.id,
            'date':self.date,
            'open':self.open,
            'close':self.close,
            'high':self.high,
            'low':self.low,
            'amount':self.amount,
            'stock':self.stock
        }

class StockDto(object):
    id: int
    date: DATE
    open: int
    close: int
    high: int
    low: int
    amount: int
    stock: str 
