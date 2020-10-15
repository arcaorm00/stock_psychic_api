from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from com_stock_api.ext.db import Base
from com_stock_api.member.member import Member

class MemberDao():
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8', encoding='utf8', echo=True)
    
    def create_table(self):
        Base.metadata.create_all(self.engine)

    def insert_member(self, memberDto):
        m = memberDto
        session.add(Member(email=m.email, password=m.password, name=m.name, geography=m.geography, age=m.age, tenure=m.tenure, balance=m.balance, has_credit=m.has_credit, is_active_member=m.is_active_member, extimated_salary=m.extimated_salary, role=m.role, exited=m.exited))
        session.commit()

    def fetch_member(self, email):
        query = session.query(Member).filter((Member.email == email))
        return query