from com_stock_api.ext.db import db

class MemberChurnPredDao:

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email)