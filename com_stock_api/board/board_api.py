from typing import List
from fastapi import APIRouter
from com_stock_api.board.board_dao import BoardDao
from com_stock_api.board.board_dto import BoardDto

class BoardrApi(object):
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router