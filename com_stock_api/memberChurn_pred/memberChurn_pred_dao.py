from com_stock_api.ext.db import db
from com_stock_api.memberChurn_pred.memberChurn_pred_dto import MemberChurnPredDto

class MemberChurnPredDao(MemberChurnPredDto):

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email)