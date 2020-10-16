from com_stock_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, FLOAT

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader
import pandas as pd

class Member(Base):

    __tablename__ = "members"
    __table_args__ = {"mysql_collate": "utf8_general_ci"}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(VARCHAR(100), unique=True, nullable=False)
    password = Column(VARCHAR(50), nullable=False)
    name = Column(VARCHAR(50))
    geography = Column(VARCHAR(30))
    gender = Column(VARCHAR(10))
    age = Column(Integer)
    tenure = Column(Integer)
    stock_qty = Column(Integer, default=0)
    balance = Column(FLOAT, default=0.0)
    has_credit = Column(Integer)
    credit_score = Column(Integer)
    is_active_member = Column(Integer, default=1)
    estimated_salary = Column(FLOAT)
    role = Column(VARCHAR(30), default='ROLE_USER')
    exited = Column(Integer, default=0)

    def __repr__(self):
        return 'Member(member_id={}, email={}, password={},'\
        'name={}, geography={}, gender={}, age={}, tenure={}, stock_qty={}, balance={},'\
        'hasCrCard={}, credit_score={}, isActiveMember={}, estimatedSalary={}, role={}, exited={}'\
        .format(self.id, self.email, self.password, self.name, self.geography, self.gender, self.age, self.tenure, self.stock_qty, self.balance, self.has_credit, self.credit_score, self.is_active_member, self.estimated_salary, self.role, self.exited)


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)
# Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
# session.add(Member(email='test@test.com', password='1234', name='test', geography='', gender='Female', age=30, tenure=0, has_credit=0, credit_score=0, estimated_salary=0))


# ---------- data를 database에 넣는 작업 ----------

# context = os.path.join(basedir, 'saved_data')
# fname = 'member_detail.csv'
# datapath = os.path.join(context, fname)
# member_data = pd.read_csv(datapath)

# for idx in range(len(member_data)):
#     # print(member_data['RowNumber'][idx], member_data['Surname'][idx])
#     m = member_data
#     session.add(Member(email=m['Email'][idx], password=m['Password'][idx], name=m['Surname'][idx], 
#     geography=m['Geography'][idx], gender=m['Gender'][idx], age=m['Age'][idx], tenure=m['Tenure'][idx], 
#     stock_qty=m['NumOfProducts'][idx], balance=m['Balance'][idx], has_credit=m['HasCrCard'][idx], 
#     credit_score=m['CreditScore'][idx], is_active_member=m['IsActiveMember'][idx], 
#     estimated_salary=m['EstimatedSalary'][idx], role=m['Role'][idx], exited=m['Exited'][idx]))


query = session.query(Member).filter((Member.name == 'test'))
print(f'query: {query}')
for m in query:
    print(m)

session.commit()