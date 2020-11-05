from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import datetime
from com_stock_api.resources.member import MemberDto

from sqlalchemy import desc, func
import pandas as pd
import json

import os
from com_stock_api.utils.file_helper import FileReader
import numpy as np
import re

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse





'''
 * @ Module Name : board.py
 * @ Description : Board
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     flask_restful, sqlalchemy
''' 




# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================





class BoardPro:

    def __init__(self):
        self.reader = FileReader()
        self.datapath = os.path.abspath(os.path.dirname(__file__))

    def process(self):
        file_data = self.get_data()
        data = self.refine_data(file_data)
        return data

    def get_data(self):
        self.reader.context = os.path.join(self.datapath, 'data')
        self.reader.fname = 'kyobo_notice.csv'
        notice_file = self.reader.csv_to_dframe()
        return notice_file
    
    @staticmethod
    def refine_data(data):
        # 컬럼명 변경
        data = data.rename({'제목': 'title', '내용': 'content', '작성일자': 'regdate'}, axis='columns')
        data = data.sort_values(by=['regdate'], axis=0)
        data['email'] = 'admin@stockpsychic.com'
        data['article_type'] = 'Notice'
        data = data.drop('url', axis=1)

        for idx in range(len(data['content'])):
            con = re.sub('<!--(.+?)-->', '', str(data['content'][idx]))
            con = con.replace('<!--', '')
            con = con.replace('교보증권', 'Stock Psychic')
            con = con.replace('\r', '<br/>')
            data['content'][idx] = con
        print(data)
        return data

    def save_data(self, data):
        self.reader.context = os.path.join(self.datapath, 'saved_data')
        self.reader.fname = 'kyobo_notice_database.csv'
        data.to_csv(self.reader.new_file(), index=False)
        print('file saved')





# =====================================================================
# =====================================================================
# ===================        modeling         =========================
# =====================================================================
# =====================================================================





class BoardDto(db.Model):

    __tablename__ = 'boards'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    article_type: str = db.Column(db.String(50), nullable=False)
    title: str = db.Column(db.String(50), nullable=False)
    content: str = db.Column(db.String(20000), nullable=False)
    regdate: datetime = db.Column(db.String(1000), default=datetime.datetime.now())

    member = db.relationship('MemberDto', back_populates='boards')
    comments = db.relationship('CommentDto', back_populates='board', lazy='dynamic')

    def __init__(self, email, article_type, title, content, regdate):
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

class BoardVo:
    id: int = 0
    email: str = ''
    article_type: str = ''
    title: str = ''
    content: str = ''
    regdate: datetime = datetime.datetime.now()







Session = openSession()
session = Session()

class BoardDao(BoardDto):

    @classmethod
    def count(cls):
        return session.query(func.count(BoardDto.id)).one()

    @classmethod
    def find_all(cls):
        sql = cls.query.order_by(cls.regdate.desc())
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, id):
        sql = cls.query.filter(cls.id.like(id))
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_member(cls, email):
        sql = cls.query.filter(cls.email.like(email))
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(board):
        db.session.add(board)
        db.session.commit()

    @staticmethod
    def insert_many():
        service = BoardPro()
        Session = openSession()
        session = Session()
        df = service.process()
        print(df.head())
        session.bulk_insert_mappings(BoardDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @classmethod
    def modify_board(cls, board):
        Session = openSession()
        session = Session()
        board = session.query(BoardDto)\
        .filter(BoardDto.id==board.id)\
        .update({BoardDto.title: board['title'], BoardDto.content: board['content']})
        session.commit()
        session.close()

    @classmethod
    def delete_board(cls, id):
        # 트랜잭션 필요

        # comments = CommentDao.find_by_boardid(id)
        # print(f'delete board comments: {comments}')
        # print(f'delete board comments: {type(comments)}')
        # for c in comments:
        #     CommentDao.delete_comment(c)
        
        # File "C:\Users\saltQ\stock_psychic_api\com_stock_api\resources\board.py", line 6, in <module>
        #     from com_stock_api.resources.comment import CommentDao
        # File "C:\Users\saltQ\stock_psychic_api\com_stock_api\resources\comment.py", line 2, in <module>
        #     from com_stock_api.resources.board import BoardDto
        # ImportError: cannot import name 'BoardDto' from 'com_stock_api.resources.board' (C:\Users\saltQ\stock_psychic_api\com_stock_api\resources\board.py)

        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        db.session.close()






# =====================================================================
# =====================================================================
# ===================        controller       =========================
# =====================================================================
# =====================================================================






parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('article_type', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('title', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('content', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('regdate', type=str, required=True, help='This field cannot be left blank')

class Board(Resource):
    
    @staticmethod
    def post(id):
        body = request.get_json()
        print(f'body: {body}')
        board = BoardDto(**body)
        BoardDao.save(board)
        return {'board': str(board.id)}, 200
    
    @staticmethod
    def get(id):
        try:
            board = BoardDao.find_by_id(id)
            if board:
                return board
        except Exception as e:
            print(e)
            return {'message': 'Board not found'}, 404

    @staticmethod
    def put(id):
        args = parser.parse_args()
        print(f'Board {args} updated')
        try:
            BoardDao.modify_board(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Board not found'}, 404
   
    @staticmethod
    def delete(id):
        try:
            BoardDao.delete_board(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Board not found'}, 404
    
class Boards(Resource):
    
    def post(self):
        b_dao = BoardDao()
        b_dao.insert_many('boards')

    def get(self):
        data = BoardDao.find_all()
        return data, 200