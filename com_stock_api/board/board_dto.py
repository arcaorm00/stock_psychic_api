from com_stock_api.ext.db import db
import datetime
from com_stock_api.member.member_dto import MemberDto

class BoardDto(db.Model):

    __tablename__ = 'boards'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    article_type: str = db.Column(db.String(50), nullable=False)
    title: str = db.Column(db.String(50), nullable=False)
    content: str = db.Column(db.String(20000), nullable=False)
    regdate: datetime = db.Column(db.String(1000), default=datetime.datetime.now())

    def __init__(self, id, email, article_type, title, content, regdate):
        self.id = id
        self.email = email
        self.article_type = article_type
        self.title = title
        self.content = content
        self.regdate = regdate

    def __repr__(self):
        return 'Board(id={}, email={}, article_type={}, title={}, content={}, regdate={})'.format(self.id, self.email, self.article_type, self.title, self.content, self.regdate)

    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'article_type': self.article_type,
            'title': self.title,
            'content': self.content,
            'regdate': self.regdate
        }

    def save(self):
        db.session.add(self)
        db.commit()

    def delete(self):
        db.session.delete(self)
        db.commit()