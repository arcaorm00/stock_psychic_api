from com_stock_api.member.member_api import MemberApi
from com_stock_api.board.board_api import BoardApi
from com_stock_api.trading.trading_api import TradingApi

def initialize_routes(api):
    api.add_resource(MemberApi, '/member')
    api.add_resource(BoardApi, '/board')
    api.add_resource(TradingApi, '/trading')