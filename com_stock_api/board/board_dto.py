from com_stock_api.ext.db import Base
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR
from com_stock_api.member.member import Member

class Board(Base):

    __tablename__ = 'boards'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey(Member.id))
    title = Column(VARCHAR(50), nullable=False)
    content = Column(VARCHAR(5000))
    regdate = Column(DateTime, default=datetime.datetime.now())

    def __repr__(self):
        return 'Board(board_id={}, member_id={}, title={}, content={}, regdate={})'.format(self.id, self.member_id, self.title, self.content, self.regdate)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'member_id': self.member_id,
            'title': self.title,
            'content': self.content,
            'regdate': self.regdate
        }

class BoardDto(object):
    id: int
    member_id: int
    title: str
    content: str
    regdate: datetime