from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.trading.trading_dao import TradingDao
from com_stock_api.trading.trading_dto import TradingDto

class TradingApi(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('kospi_stock_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('nasdaq_stock_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('stock_qty', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('price', type=float, required=True, help='This field cannot be left blank')
        parser.add_argument('trading_date', type=str, required=True, help='This field cannot be left blank')
        
    def post(self):
        data = self.parser.parse_args()
        trading = TradingDto(data['id'], data['email'], data['kospi_stock_id'], data['nasdaq_stock_id'], data['stock_qty'], data['price'], data['trading_date'])
        try:
            trading.save()
        except:
            return {'message': 'An error occured inserting the tradings'}, 500
        return trading.json(), 201
    
    def get(self, id):
        trading = TradingDao.find_by_id(id)
        if trading:
            return trading.json()
        return {'message': 'Trading not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        trading = TradingDao.find_by_id(id)

        trading.stock_qty = data['stock_qty']
        trading.price = data['price']
        trading.trading_date = data['trading_date']
        trading.save()
        return trading.json()

class Tradings(Resource):
    def get(self):
        return {'tradings': list(map(lambda trading: trading.json(), TradingDao.find_all()))}