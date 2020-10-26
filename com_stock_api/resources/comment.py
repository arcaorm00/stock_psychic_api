from com_stock_api.ext.db import db
from com_stock_api.resources.board import BoardDto
from com_stock_api.resources.member import MemberDto
import datetime

from com_stock_api.ext.db import db
import pandas as pd
import json

from flask_restful import Resource, reqparse
import datetime

class CommentDto(db.Model):

    __tablename__ = "comments"
    __table_args__ = {"mysql_collate": "utf8_general_ci"}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    board_id: int = db.Column(db.Integer, db.ForeignKey(BoardDto.id), nullable=False)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    comment: str = db.Column(db.String(500), nullable=False)
    regdate: datetime = db.Column(db.String(1000), default=datetime.datetime.now(), nullable=False)
    comment_ref: int = db.Column(db.Integer, nullable=False)
    comment_level: int = db.Column(db.Integer, nullable=False)
    comment_step: int = db.Column(db.Integer, nullable=False)

    def __init__(self, board_id, email, comment, regdate, comment_ref, comment_level, comment_step):
        self.board_id = board_id
        self.email = email
        self.comment = comment
        self.regdate = regdate
        self.comment_ref = comment_ref
        self.comment_level = comment_level
        self.comment_step = comment_step

    def __repr__(self):
        return f'id={self.id}, board_id={self.board_id}, email={self.email}, comment={self.comment}, regdate={self.regdate}, ref={self.comment_ref}, level={self.comment_level}, step={self.comment_step}'

    @property
    def json(self):
        return {
            'id': self.id,
            'board_id': self.board_id,
            'email': self.email,
            'comment': self.comment,
            'regdate': self.regdate,
            'comment_ref': self.comment_ref,
            'comment_level': self.comment_level,
            'comment_step': self.comment_step
        }

class CommentVo:
    id: int = 0
    board_id: int = 0
    email: str = ''
    comment: str = ''
    regdate: datetime = datetime.datetime.now()
    comment_ref: int = 0
    comment_level: int = 0
    comment_step: int = 0

# ref, level, step은 대댓글 기능을 위함

# ref: 최초 댓글 - 자신의 id / 대댓글 - 모댓글 ref
# level: 최초 댓글 - 0 / 대댓글 - 모댓글 level + 1
# step: 최초 댓글 - 0 / 대댓글 - 모댓글과 ref가 같은 댓글 중 모댓글보다 step이 큰 댓글 모두 step +1 이후 자신은 모댓글 step +1

'''
순서		    ref	    level	step
1		        1	    0	    0
	5	        1	    1	    1
        7	    1	    2	    2
	4	        1	    1	    3
	    6	    1	    2	    4
2		        2	    0	    0
3		        3	    0	    0

* 정렬 기준: ref asc 우선 -> step asc
'''







class CommentDao(CommentDto):

    def __init__(self):
        ...

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, comment):
        sql = cls.query.filter(cls.id.like(comment.id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_boardid(cls, comment):
        sql = cls.query.filter(cls.board_id.like(comment.board_id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(comment):
        db.session.add(comment)
        db.session.commit()

    @staticmethod
    def modify_comment(comment):
        db.session.add(comment)
        db.session.commit()

    @classmethod
    def delete_comment(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()





# =====================================================================
# =====================================================================
# ============================ controller =============================
# =====================================================================
# =====================================================================





class Comment(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('board_id', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('comment', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('regdate', type=str, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_ref', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_level', type=int, required=True, help='This field cannot be left blank')
        parser.add_argument('comment_step', type=int, required=True, help='This field cannot be left blank')
        
    def post(self):
        data = self.parser.parse_args()
        comment = CommentDto(data['id'], data['board_id'], data['email'], data['comment'], data['regdate'], data['comment_ref'], data['comment_level'], data['comment_step'])
        try:
            comment.save()
        except:
            return {'message': 'An error occured inserting the comments'}, 500
        return comment.json(), 201
    
    def get(self, id):
        comment = CommentDao.find_by_id(id)
        if comment:
            return comment.json()
        return {'message': 'Comment not found'}, 404

    def put(self, id):
        data = self.parser.parse_args()
        comment = CommentDao.find_by_id(id)

        comment.comment = data['comment']
        comment.regdate = data['regdate']
        comment.save()
        return comment.json()
    
class Comments(Resource):
    def get(self):
        return {'comments': list(map(lambda comment: comment.json(), CommentDao.find_all()))}



