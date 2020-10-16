from com_stock_api.ext.db import Base
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT
from com_stock_api.member.member import Member

class Board(Base):

    __tablename__ = 'boards'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey(Member.id))
    title = Column(VARCHAR(50), nullable=False)
    content = Column(LONGTEXT())
    regdate = Column(DateTime, default=datetime.datetime.now())

    def __repr__(self):
        return 'Board(board_id={}, member_id={}, title={}, content={}, regdate={})'.format(self.id, self.member_id, self.title, self.content, self.regdate)

engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(Board(member_id=1, title='test', content='test입니다.'))
query = session.query(Board).filter((Board.title == 'test'))
print(f'query: {query}')
for b in query:
    print(b)

session.commit()