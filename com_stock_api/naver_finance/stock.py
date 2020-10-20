from com_stock_api.ext.db import Base
from sqlalchemy import Column,Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT

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


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8',encoding='utf8', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(Stock(stock_id='1',date='2020-02-02',open='4444',close='1234',high='1233',low='222',amount='100000',stock='LGchem'))
query = session.query(Stock)
print(query)
for i in query:
    print(i)

session.commit()