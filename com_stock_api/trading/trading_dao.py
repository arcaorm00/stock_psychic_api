from com_stock_api.ext.db import db
from com_stock_api.trading.trading_dto import TradingDto
import pandas as pd
import json

class TradingDao(TradingDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, trading):
        sql = cls.query.filter_by(cls.id.like(trading.id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_email(cls, trading):
        sql = cls.query.filter_by(cls.email.like(trading.email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(trading):
        db.session.add(trading)
        db.session.commit()

    @staticmethod
    def modify_trading(trading):
        db.session.add(trading)
        db.session.commit()

    @classmethod
    def delete_trading(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
