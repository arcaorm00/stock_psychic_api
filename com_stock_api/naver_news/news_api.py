from typing import List
from fastapi import APIRouter
from com_stock_api.naver_news.news_dao import NewsDao
from com_stock_api.naver_news.news_dto import NewsDto

class NewsApi(object):

    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router