from com_stock_api.ext.db import Base
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, FLOAT
from com_stock_api.member.member import Member
# from com_stock_api.yhfinance.yhfinance import YhFinance
# from com_stock_api.naverfinance.naverfinance import NaverFinance

class Trading(Base):

    __tablename__ = "tradings"
    __table_args__ = {"mysql_collate": "utf8_general_ci"}

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey(Member.id))
    # kospi_stock_id = Column(Integer, ForeignKey(NaverFinance.id))
    # nasdaq_stock_id = Column(Integer, ForeignKey(YhFinance.id))
    price = Column(Integer, nullable=False)
    date = Column(DateTime, default=datetime.datetime.now())

    def __repr__(self):
        return 'Trading(trading_id={}, member_id={}, kospi_stock_id={}, nasdaq_stock_id={}, price={}, date={})'.format(self.id, self.member_id, self.kospi_stock_id, self.nasdaq_stock_id, self.price, self.date)


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)
# Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
# session.add(Trading(member_id=1, kospi_stock_id=1, nasdaq_stock_id=0, price=0))
query = session.query(Trading).filter((Trading.member_id == 1))
print(f'query: {query}')
for t in query:
    print(t)

session.commit()