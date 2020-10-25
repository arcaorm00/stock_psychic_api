import logging
from flask import Blueprint
from flask_restful import Api
from com_stock_api.board.board_api import Boards, Board

board = Blueprint('board', __name__, url_prefix='/api/board')
boards = Blueprint('boards', __name__, url_prefix='/api/boards')


api = Api(board)
api = Api(boards)

@board.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500