from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from com_stock_api.member.member_dao import MemberDao
from com_stock_api.member.member_dto import MemberDto, MemberVo
import json

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

class Member(Resource):

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

    @staticmethod
    def get(email):
        print(f'Member {id} added')
        try:
            member = MemberDao.find_by_email(email)
            if member:
                return member.json()
        except Exception as e:
            return {'message': 'Member not found'}, 404
    
    @staticmethod
    def update():
        args = parser.parse_args()
        print(f'Member {args["id"]} updated')
        return {'code': 0, 'message': 'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Member {args["id"]} deleted')
        return {'code': 0, 'message': 'SUCCESS'}, 200

class Members(Resource):

    def post(self):
        m_dao = MemberDao()
        m_dao.insert_many('members')

    def get(self):
        data = MemberDao.find_all()
        return data, 200
    
class Auth(Resource):

    def post(self):
        body = request.get_json()
        member = MemberDto(**body)
        MemberDao.save(member)
        email = member.email
        return {'email': str(email)}, 200
    
class Access(Resource):

    def post(self):
        args = parser.parse_args()
        member = MemberVo()
        member.email = args.email
        member.password = args.password
        print(f'email: {member.email}')
        print(f'password: {member.password}')
        data = MemberDao.login(member)
        return data[0], 200
