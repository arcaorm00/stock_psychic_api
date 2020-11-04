import logging
from flask import Blueprint
from flask_restful import Api

member = Blueprint('member', __name__, url_prefix='/api/member')
members = Blueprint('members', __name__, url_prefix='/api/members')
auth = Blueprint('auth', __name__, url_prefix='/api/auth')
access = Blueprint('access', __name__, url_prefix='/api/access')
highchurnmembers = Blueprint('highchurnmembers', __name__, url_prefix='/api/highchurnmembers')

board = Blueprint('board', __name__, url_prefix='/api/board')
boards = Blueprint('boards', __name__, url_prefix='/api/boards')
comment = Blueprint('comment', __name__, url_prefix='/api/comment')
comments = Blueprint('comments', __name__, url_prefix='/api/comments')
trading = Blueprint('trading', __name__, url_prefix='/api/trading')
tradings = Blueprint('tradings', __name__, url_prefix='/api/tradings')
recommend_stock = Blueprint('recommend_stock', __name__, url_prefix='/api/recommend-stock')
recommend_stocks = Blueprint('recommend_stocks', __name__, url_prefix='/api/recommend-stocks')



print('=============== resources / __init__.py')

api = Api(member)
api = Api(members)
api = Api(auth)
api = Api(access)
api = Api(highchurnmembers)

api = Api(board)
api = Api(boards)

api = Api(comment)
api = Api(comments)

api = Api(trading)
api = Api(tradings)

api = Api(recommend_stock)
api = Api(recommend_stocks)

@member.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500

@board.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500