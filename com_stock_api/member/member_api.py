from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from com_stock_api.member.member_dao import MemberDao
from com_stock_api.member.member_dto import MemberDto
import json

class MemberApi(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('password', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('name', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('profile', type=str, required=False, default='noimage.png')
        parser.add_argument('geography', type=str, required=False)
        parser.add_argument('gender', type=str, required=False)
        parser.add_argument('age', type=int, required=False)
        parser.add_argument('tenure', type=int, required=False)
        parser.add_argument('stock_qty', type=int, required=False)
        parser.add_argument('balance', type=float, required=False)
        parser.add_argument('has_credit', type=int, required=False)
        parser.add_argument('credit_score', type=int, required=False)
        parser.add_argument('is_active_member', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('estimated_salary', type=float, required=False)
        parser.add_argument('role', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('exited', type=int, required=True, help='This field cannot be left blank')

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Member{args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len(params) == 0:
            return 'No parameter'
        
        params_str = ''
        for key in params.key():
            params_str += 'key: {}, value: {}\n' .format(key, params[key])
        return {'code': 0, 'message': 'SUCCESS'}, 200    
    def get(self, email):
        member = MemberDao.find_by_email(email)
        if member:
            return member.json()
        return {'message': 'Member not found'}, 404

    def put(self, email):
        data = self.parser.parse_args()
        member = MemberDao.find_by_email(email)

        # 이거 뭘 하는거지?
        member.password = data['password']
        member.gender = data['gender']
        member.age = data['age']
        member.save()
        return member.json()

class Members(Resource):
    def get(self):
        return {'members': list(map(lambda member: member.json(), MemberDao.find_all()))}