from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.member import MemberDto
from com_stock_api.resources.yhfinance import YHFinanceDto
from com_stock_api.resources.korea_finance import StockDto
import datetime

import pandas as pd
import numpy as np
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse

import random
from sqlalchemy import func

import datetime
import operator

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



'''
 * @ Module Name : trading.py
 * @ Description : Trading stock
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     flask_restful
 * @ 수정일         수정자                      수정내용
 *   ------------------------------------------------------------------------
 *   2020.10.31     곽아름      데이터베이스에 insert할 자동 매수 class 추가
''' 





# =====================================================================
# =====================================================================
# =================        preprocessing         ======================
# =====================================================================
# =====================================================================





class TradingPro:

    def __init__(self):
        self.members = object
        self.kospis = object
        self.nasdaqs = object

    def hook(self):
        self.get_data()
        data = self.make_data()
        return data

    def get_data(self):
        members = pd.read_sql_table('members', engine.connect())
        kospis = pd.read_sql_table('korea_finance', engine.connect())
        nasdaqs = pd.read_sql_table('yahoo_finance', engine.connect())
        
        # kospi의 금액을 모두 20201030 현재 환율 1129.16으로 나눔
        kospis['open'] = [round(float(k)/1129.16, 4) for k in kospis['open']]
        kospis['close'] = [round(float(k)/1129.16, 4) for k in kospis['close']]
        kospis['high'] = [round(float(k)/1129.16, 4) for k in kospis['high']]
        kospis['low'] = [round(float(k)/1129.16, 4) for k in kospis['low']]

        self.members = members
        self.kospis = kospis
        self.nasdaqs = nasdaqs
    
    def make_data(self):
        kospi_ticker = self.kospis['ticker'].unique()
        nasdaq_ticker = self.nasdaqs['ticker'].unique()

        # tickers = ['051910' '011070' 'AAPL' 'TSLA']
        tickers = np.hstack([kospi_ticker, nasdaq_ticker])
        tickers[np.where(tickers == '051910')] = 'LG화학'
        tickers[np.where(tickers == '011070')] = 'LG이노텍'
        # tickers = ['LG화학' 'LG이노텍' 'AAPL' 'TSLA']

        trading_arr = []
        email = ''
        stock_type = ''
        stock_ticker = ''
        stock_qty = 0
        price = 0
        trading_date = ''

        for idx, member in self.members.iterrows():
            members_trading_qty = int(member['stock_qty'])

            if int(member['balance']) <= 0: continue
            
            # tickers의 값 중 회원의 stock_qty 수만큼 랜덤 추출
            random_ticker = random.choices(tickers, k=members_trading_qty)
            while len(set(random_ticker)) < members_trading_qty:
                random_ticker = random.choices(tickers, k=members_trading_qty)

            for tick in random_ticker:
                email = member['email']
                stock_ticker = tick
                
                if tick == 'LG화학': tick = '051910'
                if tick == 'LG이노텍': tick = '011070'

                if (self.nasdaqs['ticker'] == tick).any():
                    stock_type = 'NASDAQ'
                    nasdaq = self.nasdaqs[self.nasdaqs['ticker'] == tick]
                    
                    trading_date = ''
                    while True:
                        trading_date = random.choice(self.nasdaqs['date'])
                        if trading_date > datetime.datetime.strptime('2017-02-28', '%Y-%m-%d'):
                            break
                    high = float(nasdaq[nasdaq['date'] == trading_date]['high'])
                    low = float(nasdaq[nasdaq['date'] == trading_date]['low'])
                    price = round(random.uniform(high, low), 4)

                    stock_qty = (float(member['balance'])/members_trading_qty)/price

                if (self.kospis['ticker'] == tick).any():
                    stock_type = 'KOSPI'
                    kospi = self.kospis[self.kospis['ticker'] == tick]

                    trading_date = random.choice(self.kospis['date'])
                    high = float(kospi[kospi['date'] == trading_date]['high'])
                    low = float(kospi[kospi['date'] == trading_date]['low'])
                    price = round(random.uniform(high, low), 4)

                    stock_qty = int((float(member['balance'])/members_trading_qty)/price)
                
                result = {'email': email, 'stock_type': stock_type, 'stock_ticker': stock_ticker, 'stock_qty': int(stock_qty), 'price': price, 'trading_date': str(trading_date)}
                trading_arr.append(result)
       
        trading_df = pd.DataFrame(trading_arr)
        return trading_df






# =====================================================================
# =====================================================================
# ===================      preprocessing      =========================
# =====================================================================
# =====================================================================



class RecommendStockPreprocessing():

    def hook_process(self, members):

        isAdmin = members['email'] == 'admin@stockpsychic.com'
        members = members[~isAdmin]
        
        # 컬럼 삭제
        members = self.drop_feature(members, 'password')
        members = self.drop_feature(members, 'name')
        members = self.drop_feature(members, 'profile')
        members = self.drop_feature(members, 'role')
        members = self.drop_feature(members, 'probability_churn')
        
        # 데이터 정제
        members = self.geography_nominal(members)
        members = self.gender_nominal(members)
        members = self.age_ordinal(members)
        members = self.drop_feature(members, 'age')
        members = self.creditScore_ordinal(members)
        members = self.balance_ordinal(members)
        members = self.estimatedSalary_ordinal(members)

        return members

    # ---------------------- 데이터 정제 ----------------------
    @staticmethod
    def drop_feature(members, feature):
        members = members.drop([feature], axis=1)
        return members

    @staticmethod
    def creditScore_ordinal(members):
        members['credit_score'] = pd.qcut(members['credit_score'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members

    @staticmethod
    def geography_nominal(members):
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        members['geography'] = members['geography'].map(geography_mapping)
        return members

    @staticmethod
    def gender_nominal(members):
        gender_mapping = {'Male': 0, 'Female': 1, 'Etc.': 2}
        members['gender'] = members['gender'].map(gender_mapping)
        return members

    @staticmethod
    def age_ordinal(members):
        members['age'] = members['age'].fillna(-0.5)
        bins = [-1, 18, 25, 30, 35, 40, 45, 50, 60, np.inf] # 범위
        labels = ['Unknown', 'Youth', 'YoungAdult', 'Thirties', 'LateThirties', 'Forties', 'LateForties', 'AtferFifties', 'Senior']
        members['AgeGroup'] = pd.cut(members['age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Youth', 
            2: 'YoungAdult',
            3: 'Thirties',
            4: 'LateThirties',
            5: 'Forties',
            6: 'LateForties',
            7: 'AtferFifties',
            8: 'Senior'
        }
        age_mapping = {
            'Unknown': 0,
            'Youth': 1, 
            'YoungAdult': 2,
            'Thirties': 3,
            'LateThirties': 4,
            'Forties': 5,
            'LateForties': 6,
            'AtferFifties': 7,
            'Senior': 8
        }
        members['AgeGroup'] = members['AgeGroup'].map(age_mapping)
        return members

    @staticmethod
    def balance_ordinal(members):
        members['balance'] = pd.qcut(members['balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members

    @staticmethod
    def estimatedSalary_ordinal(members):
        members['estimated_salary'] = pd.qcut(members['estimated_salary'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return members










# =====================================================================
# =====================================================================
# =====================      similarity      ==========================
# =====================================================================
# =====================================================================



class RecommendStockModel():
    
    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'recommend_stock')

        self._this_mem = tf.placeholder(tf.float32, name='this_member')
        self._target_mem = tf.placeholder(tf.float32, name='target_member')
        self._feed_dict = {}
    
    def hook(self):
        self.substitute()
        
    def substitute(self):
        members = pd.read_sql_table('members', engine.connect())
        preprocessing = RecommendStockPreprocessing()
        refined_members = preprocessing.hook_process(members)
        
        isExitedMem = refined_members[refined_members['exited']==1].index
        refined_members = refined_members.drop(isExitedMem)
        print(f'REFINED MEMBERS: \n{refined_members}')

        refined_members.set_index(refined_members['email'], inplace=True)
        refined_members = refined_members.drop(['email'], axis=1)
        print(f'REFINED MEMBERS AFTER EMAIL INDEXING: \n{refined_members}')

        base_columns = refined_members.columns

        for email in refined_members.index:

            this_member = pd.DataFrame(refined_members.loc[email, base_columns]).T
            else_members = refined_members.loc[:, base_columns].drop(email, axis=0)

            for mem in else_members.index:

                self._feed_dict = {'this_member': this_member.loc[email, base_columns], 'target_member': else_members.loc[mem, base_columns]}
                self.create_similarity_model()

            #     if mem == '15565806@gmail.com': break          
            # if mem == '15565806@gmail.com': break
          
    def create_similarity_model(self):
        this = self._this_mem
        target = self._target_mem
        feed_dict = self._feed_dict

        _main_norm = tf.norm(this, name='this_norm')
        _row_norm = tf.norm(target, name='target_norm')

        expr_dot = tf.tensordot(this, target, 1, name='member_dot')
        expr_div = tf.divide(this, target, name='member_div')

        with tf.Session() as sess:
            _ = tf.Variable(initial_value='fake_variable')
            sess.run(tf.global_variables_initializer())

            main_m = sess.run(_main_norm, {this: feed_dict['this_member']})
            row_mem = sess.run(_row_norm, {target: feed_dict['target_member']})
            print(f'main_m: {main_m}')
            print(f'row_mem: {row_mem}')

            prod = sess.run(expr_dot, {this: feed_dict['this_member'], target: feed_dict['target_member']})
            print(f'PROD: {prod}')
            similarity = sess.run(expr_div, {this: prod, target: (main_m*row_mem)})
            print(f'SIMILARITY: {similarity}')
            
            checkpoint_path = os.path.join(self.path, 'recommend_stock_checkpoint', 'similarity.ckpt')
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_path, global_step=1000)


# if __name__ == '__main__':
#     rsmodel = RecommendStockModel()
#     rsmodel.hook()


# 위의 checkpoint 저장이 너무 느려 아래 방식으로 수정
from scipy.spatial.distance import pdist, squareform
        
class RecommendStocksWithSimilarity():

    def hook_process(self, email):
        similarity = self.similarity(email)
        sim_members = self.sortFifty(similarity)
        match_tradings = self.similarMembersTradings(sim_members, email)
        return pd.DataFrame(match_tradings)

    @staticmethod
    def similarity(email):
        members = pd.read_sql_table('members', engine.connect())
        engine.connect().close()
        isZeroBalMem = members[(members['balance'] == 0) & (members['email'] != email)].index
        members = members.drop(isZeroBalMem)

        preprocessing = RecommendStockPreprocessing()
        refined_members = preprocessing.hook_process(members)
        
        isExitedMem = refined_members[refined_members['exited']==1].index
        refined_members = refined_members.drop(isExitedMem)
        isLessBalMem = refined_members[(refined_members['balance'] <= 2) & (refined_members['email'] != email)].index
        refined_members = refined_members.drop(isLessBalMem)

        refined_members.set_index(refined_members['email'], inplace=True)
        refined_members = refined_members.drop(['email'], axis=1)

        base_index = refined_members.index
        base_columns = refined_members.columns

        row_dist = pd.DataFrame(squareform(pdist(refined_members, metric='euclidean')), columns=base_index, index=base_index)

        this_mem = row_dist[row_dist.index == email]
        this_mem = this_mem.reset_index(drop=True)

        return this_mem.to_dict(orient='records')[-1]

    @staticmethod
    def sortFifty(sim_dict):
        sim_members = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=False)[1:50]
        return sim_members

    @staticmethod
    def similarMembersTradings(sim_members, email):
        tradings = pd.read_sql_table('tradings', engine.connect())
        this_members_tradings = list(tradings[tradings['email'] == email]['stock_ticker'])

        match_tradings = pd.DataFrame(columns=('id', 'email', 'stock_type', 'stock_ticker', 'stock_qty', 'price', 'trading_date'))
        for mem, prob in sim_members:
            match_tradings = pd.concat([match_tradings, tradings[tradings['email'] == mem]])
        stocks_size = list(match_tradings.groupby('stock_ticker').size().sort_values(ascending=False).index)
        stocks_list = [{'stock_ticker':s, 
        'stock_type': str(match_tradings[match_tradings['stock_ticker'] == s]['stock_type'].unique()[0]),
        'email': email} for s in stocks_size if s not in this_members_tradings]
        return stocks_list[:2]








# =====================================================================
# =====================================================================
# ===================        modeling         =========================
# =====================================================================
# =====================================================================




class TradingDto(db.Model):

    __tablename__ = "tradings"
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    stock_type: str = db.Column(db.String(20), nullable=False)
    stock_ticker: str = db.Column(db.String(50), nullable=False)
    stock_qty: int = db.Column(db.Integer, nullable=False)
    price: float = db.Column(db.FLOAT, nullable=False)
    trading_date: str = db.Column(db.String(50), default=datetime.datetime.now())

    member = db.relationship('MemberDto', back_populates='tradings')

    def __init__(self, email, stock_type, stock_ticker, stock_qty, price, trading_date):
        self.email = email
        self.stock_type = stock_type
        self.stock_ticker = stock_ticker
        self.stock_qty = stock_qty
        self.price = price
        self.trading_date = trading_date

    def __repr__(self):
        return 'Trading(trading_id={}, email={}, stock_type={}, stock_ticker={}, stock_qty={}, price={}, trading_date={})'.format(self.id, self.email, self.stock_type, self.stock_ticker, self.stock_qty, self.price, self.trading_date)
    
    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_ticker': self.stock_ticker,
            'stock_qty': self.stock_qty,
            'price': self.price,
            'trading_date': self.trading_date
        }

class TradingVo:
    id: int = 0
    email: str = ''
    stock_type: int = 0
    stock_ticker: int = 0
    stock_qty: int = 0
    price: float = 0.0
    trading_date: datetime = datetime.datetime.now()







Session = openSession()
session = Session()

class TradingDao(TradingDto):

    def __init__(self):
        ...

    @classmethod
    def count(cls):
        return session.query(func.count(TradingDto.id)).one()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, id):
        sql = cls.query.filter(cls.id == id)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_email(cls, email):
        sql = cls.query.filter(cls.email == email)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def get_recommend_stocks(cls, email):
        rs = RecommendStocksWithSimilarity()
        df = rs.hook_process(email)
        # if len(df) <= 2:
        #     df = cls.query(func.count(TradingDto.stock_ticker).label('cnt')).groupby(TradingDto.stock_ticker).order_by('cnt'.desc())
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
        
    
    @staticmethod
    def save(trading):
        db.session.add(trading)
        db.session.commit()
        session.close()

    @staticmethod
    def insert_many():
        service = TradingPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(TradingDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @staticmethod
    def modify_trading(trading):
        Session = openSession()
        session = Session()
        trading = session.query(TradingDto)\
        .filter(TradingDto.id==trading['id'])\
        .update({TradingDto.stock_qty: trading['stock_qty'], TradingDto.price: trading['price'], TradingDto.trading_date: trading['trading_date']})
        session.commit()
        session.close()

    @classmethod
    def delete_trading(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        db.session.close()







# =====================================================================
# =====================================================================
# ============================ controller =============================
# =====================================================================
# =====================================================================




parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('email', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('stock_type', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('stock_ticker', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('stock_qty', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('price', type=float, required=True, help='This field cannot be left blank')
parser.add_argument('trading_date', type=str, required=True, help='This field cannot be left blank')


class Trading(Resource):        
        
    @staticmethod
    def post(id):
        body = request.get_json()
        print(f'body: {body}')
        trading = TradingDto(**body)
        TradingDao.save(trading)
        return {'trading': str(trading.id)}, 200
    
    @staticmethod
    def get(id):
        try:
            trading = TradingDao.find_by_id(id)
            if trading:
                return trading
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

    @staticmethod
    def put():
        args = request.get_json()
        print(f'Trading {args} updated')
        try:
            TradingDao.modify_trading(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

    @staticmethod
    def delete(id):
        try:
            TradingDao.delete_trading(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Trading not found'}, 404

class Tradings(Resource):

    def post(self):
        t_dao = TradingDao()
        t_dao.insert_many('tradings')

    def get(self, email):
        print(email)
        data = TradingDao.find_by_email(email)
        return data, 200

class TradingRecommendStock(Resource):

    def get(self, email):
        print(email)
        data = TradingDao.get_recommend_stocks(email)
        return data