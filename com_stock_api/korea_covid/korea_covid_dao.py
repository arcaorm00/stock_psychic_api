from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from com_stock_api.ext.db import Base
from com_stock_api.korea_covid.korea_covid_dto import KoreaCovid

import mysql.connector
from com_stock_api.ext.db import config

class koreaDao():

    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)

        self.connector = mysql.connector(**config)
        self.cursor = self.connector.cursor(dictionary = True)


    def create_table(self):
        Base.metadate.create_all(self.engine)

    def insert_korea_covid(self):
        session = self.session
        session.add(KoreaCovid(covid_id='1',date='2020-02-12',seoul_cases='222',seoul_death='333',total_cases='222',total_death='333'))
        session.commit()


    def fetch_korea_covid(self):
        session = self.session
        query = session.query(KoreaCovid)

    def update_korea_covid(self, db:Session, korea_covid):
        ...

    def delete_korea_covid(self, db:Session, korea_covid):
        result = db.query(KoreaCovid)
        db.delete(result)
        db.commit()