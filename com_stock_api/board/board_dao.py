from com_stock_api.ext.db import db
from com_stock_api.board.board_dto import BoardDto

class BoardDao(object):

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()

    @classmethod
    def find_by_member(cls, email):
        return cls.query.filter_by(email == email).all()
