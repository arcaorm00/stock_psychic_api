from typing import List
from fastapi import APIRouter
from com_stock_api.member.member_dao import MemberDao
from com_stock_api.member.member_dto import MemberDto

class MemberApi(object):
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        return self.router