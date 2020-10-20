from com_stock_api.ext.db import db
from com_stock_api.member.member_dto import MemberDto

class RecommendStockDto(db.Model):

    __tablename__ = 'recommend_stocks'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    stock_type: str = db.Column(db.String(50), nullable=True)
    stock_id: int = db.Column(db.Integer, nullable=False)

    def __init__(self, id, email, stock_type, stock_id):
        self.id = id
        self.email = email
        self.stock_type = stock_type
        self.stock_id = stock_id

    def __repr__(self):
        return f'id={self.id}, email={self.email}, stock_type={self.stock_type}, stock_id={self.stock_id}'

    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_id': self.stock_id
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()