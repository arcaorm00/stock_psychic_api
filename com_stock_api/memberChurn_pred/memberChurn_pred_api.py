from flask_restful import Resource, reqparse
from com_stock_api.memberChurn_pred.memberChurn_pred_dto import MemberChurnPredDto
from com_stock_api.memberChurn_pred.memberChurn_pred_dao import MemberChurnPredDao

class MemberChurnPredApi(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('prob_churn', type=float, required=True, help='This field cannot be left blank')

    def post(self):
        data = self.parser.parse_args()
        pred = MemberChurnPredDto(data['id'], data['email'], data['prob_churn'])
        try:
            pred.save()
        except:
            return {'message': 'An error occured inserting the MemberCurnPreds'}, 500
        
        return pred.json(), 201

    def get(self, id):
        pred = MemberChurnPredDao.find_by_id(id)
        if pred:
            return pred.json()

        return {'message': 'MemberChurnPrediction not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        pred = MemberChurnPredDao.find_by_id(id)

        pred.prob_churn = data['prob_churn']
        pred.save()
        return pred.json()

class MemberChurnPreds(Resource):

    def get(self):
        return {'member_churn_preds': list(map(lambda memberChurnPred: memberChurnPred.json(), MemberChurnPredDao.find_all()))}