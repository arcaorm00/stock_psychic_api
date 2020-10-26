from com_stock_api.ext.db import db
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from com_stock_api.member.member_dto import MemberDto
from com_stock_api.memberChurn_pred.memberChurn_pred_pro import MemberChurnPred

# config = {
#     'user': 'root',
#     'password': 'root',
#     'host': '127.0.0.1',
#     'port': '3306',
#     'database': 'stockdb'
# }

# charset = {'utf8': 'utf8'}
# url = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}?charset=utf8'
# engine = create_engine(url)

class MemberChurnPredDto(db.Model):
    
    __tablename__ = 'member_churn_preds'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    probability_churn: float = db.Column(db.FLOAT, nullable=False)

    def __init__(self, id, email, probability_churn):
        self.id = id
        self.email = email
        self.prob_churn = probability_churn

    def __repr__(self):
        return f'MemberChurnPred(id={self.id}, email={self.email}, prob_churn={self.prob_churn})'

    @property
    def json(self):
        return {'id': self.id, 'email': self.email, 'prob_churn': self.probability_churn}

class MemberChurnPredVo:
    id: int = 0
    email: str = ''
    probability_churn: float = 0.0

# service = MemberChurnPred()
# Session = sessionmaker(bind=engine)
# s = Session()
# df = service.hook()
# print(df.head())
# s.bulk_insert_mappings(MemberDto, df.to_dict(orient="records"))
# s.commit()
# s.close()