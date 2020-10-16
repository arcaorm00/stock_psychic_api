from typing import List
from fastapi import APIRouter
from com_stock_api.trading.trading_dao import TradingDao
from com_stock_api.trading.trading_dto import TradingDto

class TradingApi(object):
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router