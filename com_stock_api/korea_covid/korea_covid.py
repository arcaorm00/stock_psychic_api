from com_stock_api.ext.db import Base
from sqlalchemy import Column,Integer, String, ForeignKey, create_engine, DATE
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


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8',encoding='utf8',echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(KoreaCovid(covid_id='1',date='2020-02-12',seoul_cases='222',seoul_death='333',total_cases='222',total_death='333'))
query = session.query(KoreaCovid)
print(query)
for i in query:
    print(i)

session.commit()