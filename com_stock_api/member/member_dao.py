from com_stock_api.ext.db import db, openSession
from com_stock_api.member.member_pro import MemberPro
from com_stock_api.member.member_dto import MemberDto

class MemberDao(object):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        return MemberDto.query.all()

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email).first()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name == name).all()
    
    @staticmethod
    def save(member):
        db.session.add(member)
        db.session.commit()
    
    @staticmethod
    def insert_many():
        service = MemberPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(MemberDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @classmethod
    def delete_member(cls, email):
        data = cls.query.get(email)
        db.session.delete(data)
        db.session.commit()

m_dao = MemberDao()
m_dao.insert_many()