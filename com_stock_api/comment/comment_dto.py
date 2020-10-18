from com_stock_api.ext.db import db
from com_stock_api.board.board_dto import BoardDto
from com_stock_api.member.member_dto import MemberDto
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

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


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