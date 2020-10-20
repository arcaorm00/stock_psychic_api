from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from com_stock_api.ext.db import Base
from com_stock_api.naver_news.news_dto import News



import mysql.connector
from com_stock_api.ext.db import config



class NewsDao():

    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)

        self.connector = mysql.connector(**config)
        self.cursor = self.connector.cursor(dictionary = True)

    def create_table(self):
        Base.metadate.create_all(self.engine)

    def insert_naver_news(self):
        session = self.session
        session.add(News(news_id='2',date='2020-02-02',symbol='lg화학',headline='dkdif',url='hhtpt;//dkdn'))
        session.commit()

    def fetch_naver_news(self):
        session = self.session
        query = session.query(News)

    def update_naver_news(self,db:Session, naver_news):
        ...

    def delete_naver_news(self, db:Session,naver_news):
        result = db.query(News)
        db.delete(result)
        db.commit()