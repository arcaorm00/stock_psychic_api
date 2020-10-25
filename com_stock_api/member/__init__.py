import logging
from flask import Blueprint
from flask_restful import Api
from com_stock_api.member.member_api import Member

member = Blueprint('member', __name__, url_prefix='/api/member')
members = Blueprint('members', __name__, url_prefix='/api/members')
auth = Blueprint('auth', __name__, url_prefix='/api/auth')
access = Blueprint('access', __name__, url_prefix='/api/access')

api = Api(member)
api = Api(members)
api = Api(auth)
print('====== 3 ======')
api = Api(access)
print('====== 4 ======')

@member.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during user request. %s' % str(e))
    return 'An internal error occured.', 500