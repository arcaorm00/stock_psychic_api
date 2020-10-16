from com_stock_api.ext.db import db

class MemberDao():

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email).first()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name == name).all()
