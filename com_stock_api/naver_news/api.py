from flask_restful import Resource, reqparse

from com_stock_api.naver_news.dao import NewsDao
from com_stock_api.naver_news.dto import NewsDto

class News(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('neg', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('pos', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('neu', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('keywords', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('url', type=str, required=False, help='This field cannot be left blank')

    def post(self):
        data = self.parset.parse_args()
        news = NewsDto(data['datedate'],data['symbol'],data['headline'],data['neg'], data['pos'], data['neu'],data['keywords'],data['url'])
        try:
            news.save()
        except:
            return {'message':'An error occured inserting the news'}, 500
        return news.json(), 201

    def get(self,id):
        news = NewsDao.find_by_id(id)
        if news:
            return news.json()
        return {'message': 'News not found'}, 404

    def put (self, id):
        data = News.parser.parse_args()
        news = NewsDto.find_by_id(id)

        news.date = data['date']
        news.symbol = data['symbol']
        news.headline= data['headline']
        news.neg = data['neg']
        news.pos = data['pos']
        news.neu = data['neu']
        news.keywords = data['keywords']
        news.price= data['url']
        news.save()
        return news.json()

class News_(Resource):
    def get(self):
        return {'news': list(map(lambda news: news.json(), NewsDao.find_all()))}
       
