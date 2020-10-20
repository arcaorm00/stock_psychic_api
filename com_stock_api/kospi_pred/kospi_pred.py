from com_stock_api.ext.db import Base
from sqlalchemy import Column,Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT

class KospiPred(Base):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    kospi_id = Column(String(30), primary_key = True, index=True)
    date = Column(DATE)
    stock = Column(VARCHAR(30))
    price = Column(VARCHAR(30))

engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8',encoding='utf8', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(KospiPred(kospi_id='1',date='2020-02-01',stock='lg화학',price='222'))
query = session.query(KospiPred)
print(query)
for i in query:
    print(i)

session.commit()