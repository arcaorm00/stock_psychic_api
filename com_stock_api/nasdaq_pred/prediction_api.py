from flask_restful import Resource, reqparse
from com_stock_api.nasdaq_pred.prediction_dto import PredictionDto
from com_stock_api.nasdaq_pred.prediction_dao import PredictionDao

class Prediction(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('pred_price', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('past_date', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('covid_date', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('news_date', type=str, required=False, help='This field cannot be left blank')


    
    def post(self):
        data = self.parset.parse_args()
        prediction = PredictionDto(data['date'], data['ticker'],data['pred_price'], data['past_date'], data['covid_date'], data['news_date'])
        try: 
            prediction.save()
        except:
            return {'message': 'An error occured inserting the article'}, 500
        return prediction.json(), 201
    
    
    def get(self, id):
        article = PredictionDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404

    def put(self, id):
        data = Prediction.parser.parse_args()
        prediction = PredictionDao.find_by_id(id)

        prediction.title = data['title']
        prediction.content = data['content']
        prediction.save()
        return prediction.json()

class Predictions(Resource):
    def get(self):
        return {'predictions': list(map(lambda article: article.json(), PredictionDao.find_all()))}
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}