from flask_restful import Resource, reqparse
from com_stock_api.yhfinance.yhfinance_dto import YHFinanceDto
from com_stock_api.yhfinance.yhfinance_dao import YHFinanceDao

class YHFinance(Resource):


    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('open', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('close', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('adjclose', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('high', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('low', type=float, required=False, help='This field cannot be left blank')
        parser.add_argument('amount', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
    
    def post(self):
        data = self.parset.parse_args()
        stock = YHFinanceDto(data['date'], data['total_case'], data['total_death'], data['ca_cases'], data['ca_death'])
        try: 
            stock.save()
        except:
            return {'message': 'An error occured inserting the covid case'}, 500
        return stock.json(), 201
    
    
    def get(self, id):
        stock = YHFinanceDao.find_by_id(id)
        if stock:
            return stock.json()
        return {'message': 'uscovid not found'}, 404

    def put(self, id):
        data = YHFinance.parser.parse_args()
        stock = YHFinanceDao.find_by_id(id)

        stock.title = data['title']
        stock.content = data['content']
        stock.save()
        return stock.json()

class YHFinances(Resource):
    def get(self):
        return {'stock history': list(map(lambda article: article.json(), YHFinanceDao.find_all()))}
