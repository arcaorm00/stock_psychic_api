from com_stock_api.ext.db import db
from com_stock_api.comment.comment_dto import CommentDto
import pandas as pd
import json

class CommentDao(CommentDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, comment):
        sql = cls.query.filter(cls.id.like(comment.id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_boardid(cls, comment):
        sql = cls.query.filter(cls.board_id.like(comment.board_id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
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