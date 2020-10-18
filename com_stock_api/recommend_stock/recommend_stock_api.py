from typing import List
from flask_restful import Resource, reqparse
from com_stock_api.recommend_stock.recommend_stock_dao import RecommendStockDao
from com_stock_api.recommend_stock.recommend_stock_dto import RecommendStockDto

class RecommendStockApi(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('stock_type', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('stock_id', type=str, required=True, help='This field cannot be left blank')


    def post(self):
        data = self.parser.parse_args()
        recommend = RecommendStockDto(data['id'], data['email'], data['stock_type'], data['stock_id'])
        try:
            recommend.save()
        except:
            return {'message': 'An error occured inserting the RecommendStocks'}, 500
        return recommend.json(), 201
    
    def get(self, id):
        recommend = RecommendStockDao.find_by_id(id)
        if recommend:
            return recommend.json()
        return {'message': 'RecommendStocks not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        recommend = RecommendStockDao.find_by_id(id)

        recommend.stock_type = data['stock_type']
        recommend.stock_id = data['stock_id']
        recommend.save()
        return recommend.json()

class RecommendStocks(Resource):
    def get(self):
        return {'recommendStocks': list(map(lambda recommendStock: recommendStock.json(), RecommendStockDao.find_all()))}