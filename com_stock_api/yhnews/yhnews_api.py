from flask_restful import Resource, reqparse
from com_stock_api.yhnews.yhnews_dto import YHNewsDto
from com_stock_api.yhnews.yhnews_dao import YHNewsDao

class YHNews(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('headline', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('neg', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('pos', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('neu', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('compound', type=float, required=False, help='This field cannot be left blank')
    
    def post(self):
        data = self.parset.parse_args()
        news_sentiment = YHNewsDto(data['date'], data['ticker'], data['headline'], data['neg'], data['pos'], data['neu'], data['compound'])
        try: 
            news_sentiment.save()
        except:
            return {'message': 'An error occured inserting the covid case'}, 500
        return news_sentiment.json(), 201
    
    
    def get(self, id):
        news_sentiment = YHNewsDao.find_by_id(id)
        if news_sentiment:
            return news_sentiment.json()
        return {'message': 'uscovid not found'}, 404

    def put(self, id):
        data = YHNews.parser.parse_args()
        stock = YHNewsDao.find_by_id(id)

        stock.title = data['title']
        stock.content = data['content']
        stock.save()
        return stock.json()

class YHNewses(Resource):
    def get(self):
        return {'stock history': list(map(lambda article: article.json(), YHNewsDao.find_all()))}