from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, FLOAT

class Member(Base):

    __tablename__ = "members"
    __table_args__ = {"mysql_collate": "utf8_general_ci"}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(VARCHAR(100), unique=True, nullable=False)
    password = Column(VARCHAR(50), nullable=False)
    name = Column(VARCHAR(50))
    geography = Column(VARCHAR(30))
    age = Column(Integer)
    tenure = Column(Integer)
    balance = Column(FLOAT)
    has_credit = Column(Integer)
    is_active_member = Column(Integer)
    estimated_salary = Column(FLOAT)
    role = Column(VARCHAR(30))
    exited = Column(Integer)

    def __repr__(self):
        return 'Member(member_id={}, email={}, password={},'\
        'name={}, geography={}, age={}, tenure={}, balance={},'\
        'hasCrCard={}, isActiveMember={}, estimatedSalary={}, role={}, exited={}'\
        .format(self.id, self.email, self.password, self.name, self.geography, self.age, self.tenure, self.balance, self.has_credit, self.is_active_member, self.estimated_salary, self.role, self.exited)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'email': self.email,
            'password': self.password,
            'name': self.name,
            'geography': self.geography,
            'age': self.age,
            'tenure': self.tenure,
            'balance': self.balance,
            'has_credit': self.has_credit,
            'is_active_member': self.is_active_member,
            'estimated_salary': self.estimated_salary,
            'role': self.role,
            'exited': self.exited
        }

class MemberDto(object):
    id: int
    email: str
    password: str
    name: str
    geography: str
    age: int
    tenure: int
    balance: float
    has_credit: int
    is_active_member: int
    extimated_salary: float
    role: str
    exited: int