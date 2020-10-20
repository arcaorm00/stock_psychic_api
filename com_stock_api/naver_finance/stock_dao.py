from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from com_stock_api.ext.db import Base
from com_stock_api.naver_finance.stock_dto import Stock


import mysql.connector
from com_stock_api.ext.db import config

class StockDao():

    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)

        self.connector = mysql.connector(**config)
        self.cursor = self.connector.cursor(dictionary = True)

    def create_table(self):
        Base.metadate.create_all(self.engine)


    def insert_naver_finance(self):
        session = self.session
        session.add(Stock(stock_id='1',date='2020-02-02',open='4444',close='1234',high='1233',low='222',amount='100000',stock='LGchem'))
        session.commit()


    def fetch_naver_finance(self):
        session = self.session
        query = session.query(Stock)

    def update_naver_finance(self, db:Session, naver_finance):
        ...

    def delete_naver_finance(self, db:Session, naver_finance):
        result = db.query(Stock)
        db.delete(result)
        db.commit