from flask_restful import Resource, reqparse
from com_stock_api.naver_finance.dao import StockDao
from com_stock_api.naver_finance.dto import StockDto

class Stock(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()

        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('open', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('close', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('high', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('low', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('amount', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('stock', type=str, required=False, help='This field cannot be left blank')

    def post(self):
        data = self.parset.parse_args()
        stock = StockDto(data['date'],data['open'],data['close'],data['high'],data['low'],
        data['amount'],data['stock'])
        try:
            stock.save()
        except:
            return {'message':'An error occured inserting the stock'}, 500
        return stock.json(), 201

    def get(self,id):
        stock = StockDao.find_by_id(id)
        if stock:
            return stock.json()
        return {'message': 'Stock not found'}, 404

    def put (self, id):
        data = Stock.parser.parse_args()
        stock = StockDto.find_by_id(id)

        stock.open = data['date']
        stock.open = data['open']
        stock.close = data['close']
        stock.high = data['high']
        stock.low = data['low']
        stock.amount = data['amount']
        stock.stock = data['stock']
        stock.save()
        return stock.json()

class Stocks(Resource):
    def get(self):
        return {'stocks': list(map(lambda stock: stock.json(), StockDao.find_all()))}
        #return {'kospis':[kospi.json() for kospi in KospiDao.find_all()]}

