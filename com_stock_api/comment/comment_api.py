from flask_restful import Resource, reqparse
from com_stock_api.comment.comment_dao import CommentDao
from com_stock_api.comment.comment_dto import CommentDto
import datetime

class CommentApi(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('board_id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('comment', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('regdate', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_ref', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_level', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_step', type=int, required=True, help='This field cannot be left blank')
        
    def post(self):
        data = self.parser.parse_args()
        comment = CommentDto(data['id'], data['board_id'], data['email'], data['comment'], data['regdate'], data['comment_ref'], data['comment_level'], data['comment_step'])
        try:
            comment.save()
        except:
            return {'message': 'An error occured inserting the comments'}, 500
        return comment.json(), 201
    
    def get(self, id):
        comment = CommentDao.find_by_id(id)
        if comment:
            return comment.json()
        return {'message': 'Comment not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        comment = CommentDao.find_by_id(id)

        comment.comment = data['comment']
        comment.regdate = data['regdate']
        comment.save()
        return comment.json()
    
class Comments(Resource):
    def get(self):
        return {'comments': list(map(lambda comment: comment.json(), CommentDao.find_all()))}