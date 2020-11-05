from com_stock_api.resources.member import Member, Members, Auth, Access, HighChurnMembers
from com_stock_api.resources.board import Board, Boards
from com_stock_api.resources.comment import Comment, Comments, CommentMaxNum
from com_stock_api.resources.trading import Trading, Tradings
from com_stock_api.resources.recommend_stock import RecommendStock, RecommendStocks

from com_stock_api.resources.yhfinance import YHFinance, YHFinances, TeslaGraph, AppleGraph
from com_stock_api.resources.recent_news import RecentNews, AppleNews, TeslaNews
from com_stock_api.resources.investingnews import Investing, AppleSentiment, TeslaSentiment
from com_stock_api.resources.nasdaq_prediction import NasdaqPrediction, NasdaqPredictions, ApplePredGraph, TeslaPredGraph
from com_stock_api.resources.uscovid import USCovid, USCovids

from com_stock_api.resources.korea_covid import KoreaCovid,KoreaCovids
from com_stock_api.resources.kospi_pred import Kospi,Kospis,lgchem_pred,lginnotek_pred
from com_stock_api.resources.korea_finance import Stock,Stocks,lgchem,lginnotek
from com_stock_api.resources.korea_news import News,News_
from com_stock_api.resources.korea_news_recent import RNews,RNews_, lgchemNews,lginnoteknews

from com_stock_api.resources.home import Home

def initialize_routes(api):
    print('=============== route.py')

    api.add_resource(Members, '/api/members')
    api.add_resource(Member, '/api/member/<string:email>')
    api.add_resource(Auth, '/api/auth')
    api.add_resource(HighChurnMembers, '/api/highchurnmembers')
    api.add_resource(Access, '/api/access')
    api.add_resource(Boards, '/api/boards')
    api.add_resource(Board, '/api/board/<string:id>')
    api.add_resource(Comments, '/api/comments/<string:id>')
    api.add_resource(Comment, '/api/comment/<string:id>')
    api.add_resource(CommentMaxNum, '/api/commentmaxnum/<string:id>')
    api.add_resource(Tradings, '/api/tradings/<string:email>')
    api.add_resource(Trading, '/api/trading/<string:id>')
    api.add_resource(RecommendStocks, '/api/recommend-stocks')
    api.add_resource(RecommendStock, '/api/recommend-stock')

    api.add_resource(NasdaqPredictions, '/nasdaq/predictions')
    api.add_resource(ApplePredGraph, '/nasdaq/apple_pred')
    api.add_resource(TeslaPredGraph, '/nasdaq/tesla_pred')
    api.add_resource(AppleGraph, '/nasdaq/apple')
    api.add_resource(TeslaGraph, '/nasdaq/tesla')
    api.add_resource(AppleNews, '/nasdaq/apple_news')
    api.add_resource(TeslaNews, '/nasdaq/tesla_news')
    api.add_resource(AppleSentiment, '/nasdaq/apple_sentiment')
    api.add_resource(TeslaSentiment, '/nasdaq/tesla_sentiment')
    api.add_resource(USCovid, '/nasdaq/uscovid')

    api.add_resource(KoreaCovid,'/kospi/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/kospi/koreacovids')
    api.add_resource(News, '/kospi/news')
    api.add_resource(News_, '/kospi/news_') 
    api.add_resource(lgchem,'/kospi/lgchem')
    api.add_resource(lginnotek,'/kospi/lginnotek')
    api.add_resource(lgchemNews,'/kospi/lgchemNews')
    api.add_resource(lginnoteknews,'/kospi/lginnoteknews')
    api.add_resource(lgchem_pred, '/kospi/lgchem_pred')
    api.add_resource(lginnotek_pred, '/kospi/lginnotek_pred')