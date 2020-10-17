from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.trading.trading_dao import TradingDao
from com_stock_api.trading.trading_dto import TradingDto

class TradingApi(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        self.dao = TradingDao

    def get(self, id):
        trading = self.dao.find_by_id(id)

        if trading:
            return trading.json()

class Tradings(Resource):
    def get(self):
        return {'tradings': list(map(lambda trading: trading.json()))}