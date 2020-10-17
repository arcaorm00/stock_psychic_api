from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.board.board_dao import BoardDao
from com_stock_api.board.board_dto import BoardDto

class BoardrApi(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        self.dao = BoardDao
    
    def get(self, id):
        board = self.dao.find_by_id(id)

        if board:
            return board.json()

        return {'message': 'Board not found'}, 404

class Boards(Resource):
    def get(self):
        return {'boards': list(map(lambda board: board.json(), BoardDao.find_all()))}