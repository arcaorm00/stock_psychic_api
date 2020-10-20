from flask_restful import Resource, reqparse
from com_stock_api.member.member_dao import MemberDao
from com_stock_api.member.member_dto import MemberDto

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

    def post(self):
        data = self.parser.parse_args()
        member = MemberDto(data['email'], data['password'], data['name'], data['profile'], data['geography'], data['gender'], data['age'], data['tenure'], data['stock_qty'], data['balance'], data['has_credit'], data['credit_score'], data['is_active_member'], data['estimated_salary'], data['role'], data['exited'])
        try:
            member.save()
        except:
            return {'message': 'An error occured inserting the members'}, 500
        return member.json(), 201
    
    def get(self, email):
        member = MemberDao.find_by_email(email)
        if member:
            return member.json()
        return {'message': 'Member not found'}, 404

    def put(self, email):
        data = self.parser.parse_args()
        member = MemberDao.find_by_email(email)

        # 이거 뭘 하는거지?
        member.gender = data['gender']
        member.age = data['age']
        member.save()
        return member.json()

class Members(Resource):
    def get(self):
        return {'members': list(map(lambda member: member.json(), MemberDao.find_all()))}