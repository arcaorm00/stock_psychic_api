from flask import Flask
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes
from com_stock_api.member.member_api import MemberApi, Members
from com_stock_api.board.board_api import BoardApi, Boards
from com_stock_api.comment.comment_api import CommentApi, Comments
from com_stock_api.trading.trading_api import TradingApi, Tradings

app = Flask(__name__)
print('====== url ======')
print(url)
app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api = Api(app)

@app.before_first_request
def create_tables():
    db.create_all()

initialize_routes(api)

with app.app_context():
    db.create_all()