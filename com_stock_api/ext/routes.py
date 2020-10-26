from com_stock_api.resources.member import Member, Members, Auth, Access
from com_stock_api.resources.board import Board, Boards
from com_stock_api.resources.comment import Comment, Comments
from com_stock_api.resources.trading import Trading, Tradings

from com_stock_api.resources.member_churn_pred import MemberChurnPred, MemberChurnPreds
from com_stock_api.resources.recommend_stock import RecommendStock, RecommendStocks

from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

from com_stock_api.resources.prediction import Prediction, Predictions
from com_stock_api.resources.home import Home

def initialize_routes(api):
    api.add_resource(Members, '/api/members')
    api.add_resource(Member, '/api/member/<string:email>')
    api.add_resource(Auth, '/api/auth')
    print('=============== route.py')
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

    api.add_resource(Home, '/nasdaq')
    api.add_resource(Prediction, '/nasdaq/prediction')
    api.add_resource(Predictions, '/nasdaq/predictions')