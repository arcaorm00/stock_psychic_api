from com_stock_api.member.member_api import Member, Members, Auth, Access
from com_stock_api.board.board_api import Board, Boards
from com_stock_api.comment.comment_api import Comment, Comments
from com_stock_api.trading.trading_api import Trading, Tradings

from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPred, MemberChurnPreds
from com_stock_api.recommend_stock.recommend_stock_api import RecommendStock, RecommendStocks

from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

from com_stock_api.nasdaq_pred.prediction_api import Prediction, Predictions

def initialize_routes(api):
    api.add_resource(Members, '/api/members')
    api.add_resource(Member, '/api/member/<string:email>')
    api.add_resource(Auth, '/api/auth')
    api.add_resource(Access, '/api/access')
    api.add_resource(Boards, '/api/boards')
    api.add_resource(Board, '/api/board/<string:id>')
    api.add_resource(Comments, '/api/comments')
    api.add_resource(Comment, '/api/comment/<string:id>')
    api.add_resource(Tradings, '/api/tradings')
    api.add_resource(Trading, '/api/trading/<string:id>')
    api.add_resource(MemberChurnPreds, '/api/member-churn-preds')
    api.add_resource(MemberChurnPred, '/api/member-churn-preds')
    api.add_resource(RecommendStocks, '/api/recommend-stocks')
    api.add_resource(RecommendStock, '/api/recommend-stocks')

    api.add_resource(KoreaCovid,'/api/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/api/koreacovids')
    api.add_resource(Kospi,'/api/kospi/<string:id>')
    api.add_resource(Kospis,'/api/kospis')
    api.add_resource(Stock,'/api/stock/<string:id>')
    api.add_resource(Stocks,'/api/stocks')
    api.add_resource(News,'/api/news/<string:id>')
    api.add_resource(News_,'/api/news_')

    api.add_resource(Prediction, '/api/prediction')