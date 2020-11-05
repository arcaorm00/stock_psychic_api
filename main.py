from flask import Flask, request
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes

from com_stock_api.resources.member import MemberDao
from com_stock_api.resources.board import BoardDao
from com_stock_api.resources.comment import CommentDao
from com_stock_api.resources.trading import TradingDao
from com_stock_api.resources.recommend_stock import RecommendStockDao

from com_stock_api.resources.nasdaq_prediction import NasdaqPredictionDao
from com_stock_api.resources.uscovid import USCovidDao
from com_stock_api.resources.yhfinance import YHFinanceDao
from com_stock_api.resources.investingnews import InvestingDao
from com_stock_api.resources.recent_news import RecentNewsDao

from com_stock_api.resources.korea_news import NewsDao
from com_stock_api.resources.korea_covid import KoreaDao
from com_stock_api.resources.korea_finance import StockDao
from com_stock_api.resources.korea_news_recent import RNewsDao
# from com_stock_api.resources.kospi_pred import KospiDao

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
    if count == (0,):
        MemberDao.insert_many()

with app.app_context():
    count = BoardDao.count()
    print(f'Boards Total Count is {count}')
    if count == (0,):
        BoardDao.insert_many()


with app.app_context():
    count1 = USCovidDao.count()
    print(f'US Covid case Total Count is {count1[0]}')
    if count1[0] == 0:
        USCovidDao.bulk()

    count2 = YHFinanceDao.count()
    print(f'YHFinance data Total Count is {count2[0]}')
    if count2[0] == 0:
        YHFinanceDao.bulk()

    count3 = InvestingDao.count()
    print(f'Stock news Total Count is {count3[0]}')
    if count3[0] == 0:
        InvestingDao.bulk()

    count4 = RecentNewsDao.count()
    print(f'Recent news Total Count is {count4[0]}')
    if count4[0] == 0:
        RecentNewsDao.bulk()

    count5 = NasdaqPredictionDao.count()
    print(f'Nasdap Prediction Total Count is {count5}')
    if count5 == 0:
        NasdaqPredictionDao.bulk()



with app.app_context():
    news_count = NewsDao.count()
    print(f'****** News Total Count is {news_count} *******')
    if news_count[0] == 0:
        #NewsDao()
        n = NewsDao()
        n.bulk()

    covid_count = KoreaDao.count()
    print(f'***** Covid Count is {covid_count} *******')
    if covid_count[0] == 0:
        #KoreaDao().bulk()
        k = KoreaDao()
        k.bulk()

    recent_stock_count = StockDao.count()
    print(f'****Stock Count is {recent_stock_count} ****')
    if recent_stock_count[0] == 0:
        #StockDao.bulk()
        rs = StockDao()
        rs.bulk()
    
    recent_news_count = RNewsDao.count()
    print(f'******* Recent News Count is {recent_news_count}*****')
    if recent_news_count[0] == 0:
        #RNewsDao.bulk()
        rn = RNewsDao()
        rn.bulk()

    # pred_count = KospiDao.count()
    # print(f'***** Pred Count is {pred_count} *********')
    # if pred_count[0] == 0:
    #     #KospiDao.bulk()
    #     kp = KospiDao()
    #     kp.bulk()



with app.app_context():
    count = TradingDao.count()
    print(f'Tradings Total Count is {count}')
    if count == (0,):
        TradingDao.insert_many()

initialize_routes(api)