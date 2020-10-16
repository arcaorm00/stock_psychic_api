from com_stock_api.ext.db import db

class CommentDao():

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()

    @classmethod
    def find_by_boardid(cls, board_id):
        return cls.query.filter_by(board_id == board_id).all()