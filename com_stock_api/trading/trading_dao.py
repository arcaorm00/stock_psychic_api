from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from com_stock_api.ext.db import Base
from com_stock_api.trading.trading_dto import Trading

class TradingDao():

    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8', encoding='utf8', echo=True)

    def create_table(self):
        Base.metadata.create_all(self.engine)

    def insert_trading(self):
        session = self.session
        session.add(Trading(member_id=1, kospi_stock_id=1, nasdaq_stock_id=0, stock_qty=1, price=0))
        session.commit()
    
    def fetch_trading(self):
        session = self.session
        query = session.query(Trading).filter((Trading.id == 1))
        return query[0]

    def fetch_all_trading(self):
        ...

    def update_trading(self):
        ...

    def delete_trading(self):
        ...