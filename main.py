from flask import Flask, request
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes

# from com_stock_api.member import member
from com_stock_api.member.member_api import Member, Members
from com_stock_api.board.board_api import Board, Boards
from com_stock_api.comment.comment_api import Comment, Comments
from com_stock_api.trading.trading_api import Trading, Tradings
from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPred, MemberChurnPreds
from com_stock_api.recommend_stock.recommend_stock_api import RecommendStock, RecommendStocks
from com_stock_api.member import member
from com_stock_api.board import board

from com_stock_api.nasdaq_pred.prediction_api import Prediction, Predictions
from com_stock_api.us_covid.us_covid_api import USCovid, USCovids
from com_stock_api.yhfinance.yhfinance_api import YHFinance, YHFinances
from com_stock_api.yhnews.yhnews_api import YHNews, YHNewses

from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(member)
app.register_blueprint(board)

print('====== url ======')
print(url)

app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api = Api(app)

# @app.before_first_request
# def create_tables():
#     db.create_all()

initialize_routes(api)

with app.app_context():
    db.create_all()

@app.route('/api')
def connect_test():
    return {'message': 'connect success'}

# ===================== Member
# @app.route('/api/member/insert', methods=['POST'])
# def insert_member():
#     result = Member.post()
#     print(result)
#     return result

# @app.route('/api/members/list')
# def list_members():
#     members = Members()
#     members_list = members.get()
#     print(members_list)
#     return members_list

# @app.route('/api/member/get-by-email', methods=['POST'])
# def get_members_by_email():
#     email = request.get_json()['email']
#     print(email)
#     member = Member.get(email)
#     return member

# ===================== Board
# @app.route('/api/boards/insert')
# def insert_article():
#     result = Board.post()
#     return result

# @app.route('/api/boards/list')
# def list_articles():
#     boards_list = Boards.get()
#     print(boards_list)
#     return boards_list

# @app.route('/api/boards/detail')
# def get_article_by_id(id):
#     board = Board.get(id)
#     return board

# # ===================== trading
# @app.route('/api/trading/insert')
# def insert_trading():
#     result = Trading.post()
#     return result

# @app.route('/api/trading/get-by-email')
# def get_tradings_by_email(email):
#     trading_list = Tradings.get_by_email(email)
#     print(trading_list)
#     return trading_list