from flask import Flask, request
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes

# from com_stock_api.member import member
from com_stock_api.resources.member import MemberDao
from com_stock_api.resources.board import BoardDao
from com_stock_api.resources.member_churn_pred import MemberChurnPredDao

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
CORS(app, resources={r'/api/*': {"origins": "*"}})

# app.register_blueprint(member)
# app.register_blueprint(board)

print('====== url ======')
print(url)

app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api = Api(app)

with app.app_context():
    db.create_all()

with app.app_context():
    count = MemberDao.count()
    print(f'Members Total Count is {count}')
    if count == 0:
        MemberDao.insert_many()

with app.app_context():
    count = BoardDao.count()
    print(f'Boards Total Count is {count}')
    if count == 0:
        BoardDao.insert_many()

with app.app_context():
    count = MemberChurnPredDao.count()
    print(f'MemberChurnPredictions Total Count is {count}')
    if count == 0:
        MemberChurnPredDao.insert_many()

initialize_routes(api)