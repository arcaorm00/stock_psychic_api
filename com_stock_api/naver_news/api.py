from flask_restful import Resource, reqparse

from com_stock_api.naver_news.dao import NewsDao
from com_stock_api.naver_news.dto import NewsDto

class News(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()

        parser.add_argument('news_date', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('symbol', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('url', type=str, required=False, help='This field cannot be left blank')

    def post(self):
        data = self.parset.parse_args()
        news = NewsDto(data['news_date'],data['symbol'],data['headline'],data['url'])
        try:
            news.save()
        except:
            return {'message':'An error occured inserting the news'}, 500
        return news.json(), 201

    def get(self,news_date):
        news = NewsDao.find_by_id(news_date)
        if news:
            return news.json()
        return {'message': 'News not found'}, 404

    def put (self, news_date):
        data = News.parser.parse_args()
        news = NewsDto.find_by_id(news_date)

        news.stock = data['symbol']
        news.price= data['headline']
        news.price= data['url']
        news.save()
        return news.json()

class News_(Resource):
    def get(self):
        return {'news': list(map(lambda news: news.json(), NewsDao.find_all()))}
       
