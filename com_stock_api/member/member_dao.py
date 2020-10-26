from com_stock_api.ext.db import db, openSession
from com_stock_api.member.member_dto import MemberDto
from com_stock_api.member.member_pro import MemberPro
import pandas as pd
import json

class MemberDao(MemberDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_email(cls, member):
        sql = cls.query.filter(cls.email.like(member.email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_name(cls, member):
        sql = cls.query.filter(cls.name.like(member.name))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def login(cls, member):
        sql = cls.query.filter(cls.email.like(member.email))\
            .filter(cls.password.like(member.password))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print('=======================================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(member):
        db.session.add(member)
        db.session.commit()

    @staticmethod
    def insert_many():
        service = MemberPro()
        session = openSession()
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
# MemberDao.insert_many()