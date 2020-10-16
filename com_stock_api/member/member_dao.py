from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from com_stock_api.ext.db import Base
from com_stock_api.member.member_dto import Member

class MemberDao():
    
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8', encoding='utf8', echo=True)
    
    def create_table(self):
        Base.metadata.create_all(self.engine)

    def insert_member(self, memberDto):
        session = self.session
        # m = memberDto
        # session.add(Member(email=m.email, password=m.password, name=m.name, geography=m.geography, age=m.age, tenure=m.tenure, balance=m.balance, has_credit=m.has_credit, is_active_member=m.is_active_member, extimated_salary=m.extimated_salary, role=m.role, exited=m.exited))
        session.add(Member(email='test@test.com', password='1234', name='test', geography='', age=30, tenure=0, balance=0.0, has_credit=0, is_active_member=1, estimated_salary=0, role='ROLE_USER', exited=0))
        session.commit()

    def fetch_member(self, email):
        session = self.session
        query = session.query(Member).filter((Member.email == email))
        return query[0]

    def fetch_all_members(self):
        query = session.query(Member)
        return query

    def update_member(self):
        ...

    def delete_member(self):
        ...