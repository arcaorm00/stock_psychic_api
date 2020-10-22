from com_stock_api.ext.db import db, openSession
from com_stock_api.board.board_dto import BoardDto
from com_stock_api.board.board_pro import BoardPro

class BoardDao(BoardDto):
    
    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()

    @classmethod
    def find_by_member(cls, email):
        return cls.query.filter_by(email == email).all()

    @staticmethod
    def save(board):
        db.session.add(board)
        db.session.commit()

    @staticmethod
    def insert_many():
        service = BoardPro()
        Session = openSession()
        session = Session()
        df = service.process()
        print(df.head())
        session.bulk_insert_mappings(BoardDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @staticmethod
    def modify_board(board):
        db.session.add(board)
        db.commit()

    @classmethod
    def delete_board(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()

# b_dao = BoardDao()
# b_dao.insert_many()