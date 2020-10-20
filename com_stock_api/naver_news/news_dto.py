from com_stock_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import VARCHAR

class News(Base):
    
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    news_id = Column(String(30), primary_key = True, index = True)
    date = Column(DATE)
    symbol = Column(VARCHAR(30))
    headline = Column(VARCHAR(30))
    url = Column(VARCHAR(30))

    def __repr__(self):
        return 'News(news_id={},date={},symbol={},headline={},url={}'\
            .format(self.id,self.date,self.symbol,self.headline,self.url)



    @property
    def serialize(self):
        return{
            'news_id':self.id,
            'date':self.date,
            'symbol':self.symbol,
            'headline':self.headline,
            'url':self.url
        }

class NewsDto(object):
    id: int
    date : DATE
    symbol : str
    headline : str
    headline : str
    url : str