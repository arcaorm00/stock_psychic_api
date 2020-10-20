from flask_restful import Resource
from flask import Response, jsonify
from com_stock_api.kospi_pred.kospi_pred_dao import KospiDao

class KospiApi(Resource):

    def __init__(self):
        self.dao = KospiDao

    def get(self):
        kospi = self.dao.select_kospi()
        return jsonify(kospi[0])