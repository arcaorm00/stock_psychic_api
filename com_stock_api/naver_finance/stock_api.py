from typing import List
from fastapi import APIRouter
from com_stock_api.naver_finance.stock_dao import StockDao
from com_stock_api.naver_finance.stock_dto import StockDto

class StocksApi(object):
    
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router