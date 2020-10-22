from com_stock_api.member.member_api import MemberApi
from com_stock_api.board.board_api import BoardApi
from com_stock_api.comment.comment_api import CommentApi
from com_stock_api.trading.trading_api import TradingApi

from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPredApi
from com_stock_api.recommend_stock.recommend_stock_api import RecommendStockApi

from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

def initialize_routes(api):
    api.add_resource(MemberApi, '/api/members')
    api.add_resource(BoardApi, '/api/boards')
    api.add_resource(CommentApi, '/api/comments')
    api.add_resource(TradingApi, '/api/tradings')
    api.add_resource(MemberChurnPredApi, '/api/member-churn-preds')
    api.add_resource(RecommendStockApi, '/api/recommend-stocks')

    api.add_resource(KoreaCovid,'/api/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/api/koreacovids')
    api.add_resource(Kospi,'/api/kospi/<string:id>')
    api.add_resource(Kospis,'/api/kospis')
    api.add_resource(Stock,'/api/stock/<string:id>')
    api.add_resource(Stocks,'/api/stocks')
    api.add_resource(News,'/api/news/<string:id>')
    api.add_resource(News_,'/api/news_')