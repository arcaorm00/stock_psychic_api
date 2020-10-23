from com_stock_api.ext.db import db
from com_stock_api.comment.comment_dto import CommentDto

class CommentDao(CommentDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()

    @classmethod
    def find_by_boardid(cls, board_id):
        return cls.query.filter_by(board_id == board_id).all()
    
    @staticmethod
    def save(comment):
        db.session.add(comment)
        db.session.commit()

    @staticmethod
    def modify_comment(comment):
        db.session.add(comment)
        db.session.commit()

    @classmethod
    def delete_comment(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()