from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.comment.comment_dao import CommentDao
from com_stock_api.comment.comment_dto import CommentDto

class CommentApi(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        self.dao = CommentDao

    def get(self, id):
        comment = self.dao.find_by_id()

        if comment:
            return comment.json()

        return {'massage': 'Comment not found'}, 404
    
class Comments(Resource):
    def get(self):
        return {'comments': list(map(lambda comment: comment.json(), CommentDao.find_all()))}