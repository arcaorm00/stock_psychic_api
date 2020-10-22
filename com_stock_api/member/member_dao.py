from com_stock_api.ext.db import db, openSession
from com_stock_api.member.member_dto import MemberDto
from com_stock_api.member.member_pro import MemberPro

class MemberDao(MemberDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email == email).first()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name == name).all()

    def login(cls, member):
        return cls.query.filter(cls.email == member.email)\
        .filter(cls.password == member.password).first()

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
    
    @staticmethod
    def modify_member(member):
        db.session.add(member)
        db.session.commit()
    
    @classmethod
    def delete_member(cls, email):
        data = cls.query.get(email)
        db.session.delete(data)
        db.session.commit()

# m_dao = MemberDao()
# m_dao.insert_many()