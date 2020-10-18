from com_stock_api.ext.db import db
from com_stock_api.member.member_dto import MemberDto

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

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()