from com_stock_api.ext.db import db
from com_stock_api.trading.trading_dto import TradingDto

class TradingDao(TradingDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()
    
    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email).all()

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
