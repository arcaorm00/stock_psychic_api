#from flask_restful import Resource
#from flask import Response, jsonify
from typing import List
from fastapi import APIRouter
from com_stock_api.korea_covid.korea_covid_dao import koreaDao
from com_stock_api.korea_covid.korea_covid_dto import KoreaDto

class KoreaApi(object):

    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router