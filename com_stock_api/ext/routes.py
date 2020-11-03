from com_stock_api.resources.member import Member, Members, Auth, Access, HighChurnMembers
from com_stock_api.resources.board import Board, Boards
from com_stock_api.resources.comment import Comment, Comments, CommentMaxNum
from com_stock_api.resources.trading import Trading, Tradings

# from com_stock_api.resources.recommend_stock import RecommendStock, RecommendStocks

from com_stock_api.resources.prediction import Prediction, Predictions
from com_stock_api.resources.home import Home

def initialize_routes(api):
    api.add_resource(Members, '/api/members')
    api.add_resource(Member, '/api/member/<string:email>')
    api.add_resource(Auth, '/api/auth')
    api.add_resource(HighChurnMembers, '/api/highchurnmembers')
    print('=============== route.py')
    api.add_resource(Access, '/api/access')
    api.add_resource(Boards, '/api/boards')
    api.add_resource(Board, '/api/board/<string:id>')
    api.add_resource(Comments, '/api/comments/<string:id>')
    api.add_resource(Comment, '/api/comment/<string:id>')
    api.add_resource(CommentMaxNum, '/api/commentmaxnum/<string:id>')
    api.add_resource(Tradings, '/api/tradings/<string:email>')
    api.add_resource(Trading, '/api/trading/<string:id>')
    # api.add_resource(RecommendStocks, '/api/recommend-stocks')
    # api.add_resource(RecommendStock, '/api/recommend-stock')

    api.add_resource(Home, '/nasdaq')
    api.add_resource(Prediction, '/nasdaq/prediction')
    api.add_resource(Predictions, '/nasdaq/predictions')