from com_stock_api.ext.db import db


class USCovidDao():

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_date(cls, date):
        return cls.query.filer_by(date == date).all()