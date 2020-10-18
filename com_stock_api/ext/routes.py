from com_stock_api.member.member_api import MemberApi
from com_stock_api.board.board_api import BoardApi
from com_stock_api.comment.comment_api import CommentApi
# from com_stock_api.trading.trading_api import TradingApi

def initialize_routes(api):
    api.add_resource(MemberApi, '/api/members')
    api.add_resource(BoardApi, '/api/boards')
    api.add_resource(CommentApi, '/api/comments')
    # api.add_resource(TradingApi, '/api/tradings')