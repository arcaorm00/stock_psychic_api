from com_stock_api.ext.db import db
from com_stock_api.recommend_stock.recommend_stock_dto import RecommendStockDto
import pandas as pd
import json

class RecommendStockDao(RecommendStockDto):

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, recommend):
        sql = cls.query.filter(cls.id.like(recommend.id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, recommend):
        sql = cls.query.filter(cls.email.like(recommend.email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))


    @staticmethod
    def save(recommend_stock):
        db.session.add(recommend_stock)
        db.session.commit()
    
    @staticmethod
    def modify_recommend_stock(recommend_stock):
        db.session.add(recommend_stock)
        db.session.commit()

    @classmethod
    def delete_recommend_stock(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()