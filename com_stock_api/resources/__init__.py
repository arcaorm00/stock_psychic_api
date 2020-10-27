import logging
from flask import Blueprint
from flask_restful import Api

member = Blueprint('member', __name__, url_prefix='/api/member')
members = Blueprint('members', __name__, url_prefix='/api/members')
auth = Blueprint('auth', __name__, url_prefix='/api/auth')
access = Blueprint('access', __name__, url_prefix='/api/access')

board = Blueprint('board', __name__, url_prefix='/api/board')
boards = Blueprint('boards', __name__, url_prefix='/api/boards')


print('=============== resources / __init__.py')

api = Api(member)
api = Api(members)
api = Api(auth)
api = Api(access)

api = Api(board)
api = Api(boards)

@member.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500

@board.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500