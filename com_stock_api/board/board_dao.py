from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from com_stock_api.ext.db import Base
from com_stock_api.board.board_dto import Board

class BoardDao():

    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8', encoding='utf8', echo=True)

    def create_table(self):
        Base.metadate.create_all(self.engine)

    def insert_board(self):
        session = self.session
        session.add(Board(member_id=1, title='test', content='test입니다.'))
        session.commit()
    
    def fetch_board(self):
        session = self.session
        query = session.query(Board).filter((Board.id == 1))
    
    def fetch_all_boards(self):
        ...

    def update_board(self):
        ...
    
    def delete_board(self):
        ...
