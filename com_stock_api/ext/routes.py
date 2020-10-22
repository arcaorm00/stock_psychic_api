from com_stock_api.member.member_api import MemberApi, Members
from com_stock_api.board.board_api import BoardApi, Boards
from com_stock_api.comment.comment_api import CommentApi, Comments
from com_stock_api.trading.trading_api import TradingApi, Tradings

from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPredApi, MemberChurnPreds
from com_stock_api.recommend_stock.recommend_stock_api import RecommendStockApi, RecommendStocks

from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

from com_stock_api.nasdaq_pred.prediction_api import Prediction, Predictions

def initialize_routes(api):
    api.add_resource(Members, '/api/members')
    api.add_resource(MemberApi, '/api/member/get-by-email/<string:email>')
    api.add_resource(Boards, '/api/boards')
    api.add_resource(BoardApi, '/api/board/<string:id>')
    api.add_resource(Comments, '/api/comments')
    api.add_resource(CommentApi, '/api/comment/<string:id>')
    api.add_resource(Tradings, '/api/tradings')
    api.add_resource(TradingApi, '/api/trading/<string:id>')
    api.add_resource(MemberChurnPreds, '/api/member-churn-preds')
    api.add_resource(MemberChurnPredApi, '/api/member-churn-preds')
    api.add_resource(RecommendStocks, '/api/recommend-stocks')
    api.add_resource(RecommendStockApi, '/api/recommend-stocks')

    api.add_resource(KoreaCovid,'/api/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/api/koreacovids')
    api.add_resource(Kospi,'/api/kospi/<string:id>')
    api.add_resource(Kospis,'/api/kospis')
    api.add_resource(Stock,'/api/stock/<string:id>')
    api.add_resource(Stocks,'/api/stocks')
    api.add_resource(News,'/api/news/<string:id>')
    api.add_resource(News_,'/api/news_')

    api.add_resource(Prediction, '/api/prediction')