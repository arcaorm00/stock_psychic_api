from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.board.board_dao import BoardDao
from com_stock_api.board.board_dto import BoardDto

class BoardApi(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('article_type', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('title', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('content', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('regdate', type=str, required=True, help='This field cannot be left blank')
        
    def post(self):
        data = self.parser.parse_args()
        board = BoardDto(data['id'], data['email'], data['title'], data['content'], data['regdate'])
        try:
            board.save()
        except:
            return {'message': 'An error occured inserting the articls'}, 500
        return board.json(), 201
    
    def get(self, id):
        board = BoardDao.find_by_id(id)
        if board:
            return board.json()
        return {'message': 'Board not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        board = BoardDao.find_by_id(id)

        board.article_type = data['article_type']
        board.title = data['title']
        board.content = data['content']
        board.regdate = data['regdate']
        board.save()
        return board.json()
    
class Boards(Resource):
    def get(self):
        return {'boards': list(map(lambda board: board.json(), BoardDao.find_all()))}