from com_stock_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import VARCHAR


class KospiPred(Base):
    
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    kospi_id = Column(String(30), primary_key = True, index=True)
    date = Column(DATE)
    stock = Column(VARCHAR(30))
    price = Column(VARCHAR(30))

    def __repr__(self):
        return 'KospiPred(kospi_id={},date={},stock={},price={})'.format(self.id,self.data,\
            self.stock,self.price)


    @property
    def serialize(self):
        return {
            'kospi_id':self.id,
            'date':self.date,
            'stock':self.stock,
            'price':self.price
        }

class KospiDto(object):
    id : int
    date: DATE
    stock: int
    price: int