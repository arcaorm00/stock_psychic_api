from com_stock_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import VARCHAR

class KoreaCovid(Base):
    
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    covid_id = Column(String(30),primary_key = True, index = True )
    # kospi_id = Column(Integer, ForeignKey(KospiPred.id))
    # stock_id = Column(Integer, ForeignKey(Stock.id))
    # news_id = Column(Integer, ForeignKey(News.id))
    date = Column(DATE)
    seoul_cases = Column(VARCHAR(30))
    seoul_death = Column(VARCHAR(30))
    total_cases = Column(VARCHAR(30))
    total_death = Column(VARCHAR(30))

    def __repr__(self):
        return 'KoreaCovid(covid_id={}, date={},seoul_cases={},seoul_death={},\
            total_cases={},total_death={})'.format(self.id,self.data,self.seoul_cases,\
                self.seoul_death,self.total_cases,self.total_death)

    @property
    def serialize(self):
        return {
            'covid_id':self.id,
            'date':self.date,
            'seoul_cases':self.seoul_cases,
            'seoul_death':self.seoul_death,
            'total_cases':self.total_cases,
            'total_death':self.total_death
        }

class KoreaDto(object):
    id: int
    date: DATE
    seoul_cases: int
    seoul_death: int
    total_cases: int
    total_death: int