from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from com_stock_api.ext.db import Base
from com_stock_api.korea_covid.korea_covid_dto import KospiPred

import mysql.connector
from com_stock_api.ext.db import config

class KospiDao():
    
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)

        self.connector = mysql.connector(**config)
        self.cursor = self.connector.cursor(dictionary = True)

    def create_table(self):
        Base.metadate.create_all(self.engine)

    def insert_kospi_pred(self):
        session = self.session
        session.add(KospiPred(kospi_id='1',date='2020-02-01',stock='lg화학',price='222'))
        session.commit()

    def fetch_kospi_pred(self):
        session =self.session
        query = session.query(KospiPred)

    def update_kospi_pred(self,db:Session, kospi_pred):
        ...

    def delete_kospi_pred(self, db:Session, kospi_pred):
        result = db.query(KospiPred)
        db.delete(result)
        db.commit()