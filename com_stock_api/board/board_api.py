from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from com_stock_api.board.board_dao import BoardDao
from com_stock_api.board.board_dto import BoardDto
import json

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('article_type', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('title', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('content', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('regdate', type=str, required=True, help='This field cannot be left blank')

class Board(Resource):
    
    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Board {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
    
    @staticmethod
    def get(id):
        try:
            board = BoardDao.find_by_id(id)
            # print(board)
            if board:
                return board
        except Exception as e:
            print(e)
            return {'message': 'Board not found'}, 404

    @staticmethod
    def update():
        args = parser.parse_args()
        print(f'Board {args["id"]} updated')
        return {'code': 0, 'message': 'SUCCESS'}, 200
   
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Board {args["id"]} deleted')
        return {'code': 0, 'message': 'SUCCESS'}, 200
    
class Boards(Resource):
    
    def post(self):
        b_dao = BoardDao()
        b_dao.insert_many('boards')

    def get(self):
        data = BoardDao.find_all()
        return data, 200