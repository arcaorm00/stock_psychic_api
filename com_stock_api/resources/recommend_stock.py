from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.member import MemberDto

import pandas as pd
import json

from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse

import numpy as np
import math
import operator

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


'''
 * @ Module Name : recommend_stock.py
 * @ Description : Recommend Stock
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 곽아름
 * @ special reference libraries
 *     flask_restful
 * @ 수정일         수정자                      수정내용
 *   ------------------------------------------------------------------------
 *   2020.11.03     곽아름      멤버간 유사도 측정 및 추천 종목 추출
''' 




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





# class RecommendStockModel():
    
#     def __init__(self):
#         self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'recommend_stock')

#         self._this_mem = tf.placeholder(tf.float32, name='this_member')
#         self._target_mem = tf.placeholder(tf.float32, name='target_member')
#         self._feed_dict = {}
    
#     def hook(self):
#         self.substitute()
        
#     def substitute(self):
#         members = pd.read_sql_table('members', engine.connect())
#         preprocessing = RecommendStockPreprocessing()
#         refined_members = preprocessing.hook_process(members)
        
#         isExitedMem = refined_members[refined_members['exited']==1].index
#         refined_members = refined_members.drop(isExitedMem)
#         print(f'REFINED MEMBERS: \n{refined_members}')

#         refined_members.set_index(refined_members['email'], inplace=True)
#         refined_members = refined_members.drop(['email'], axis=1)
#         print(f'REFINED MEMBERS AFTER EMAIL INDEXING: \n{refined_members}')

#         base_columns = refined_members.columns

#         for email in refined_members.index:

#             this_member = pd.DataFrame(refined_members.loc[email, base_columns]).T
#             else_members = refined_members.loc[:, base_columns].drop(email, axis=0)

#             for mem in else_members.index:

#                 self._feed_dict = {'this_member': this_member.loc[email, base_columns], 'target_member': else_members.loc[mem, base_columns]}
#                 self.create_similarity_model()

#             #     if mem == '15565806@gmail.com': break
            
#             # if mem == '15565806@gmail.com': break
          
#     def create_similarity_model(self):
#         this = self._this_mem
#         target = self._target_mem
#         feed_dict = self._feed_dict

#         _main_norm = tf.norm(this, name='this_norm')
#         _row_norm = tf.norm(target, name='target_norm')

#         expr_dot = tf.tensordot(this, target, 1, name='member_dot')
#         expr_div = tf.divide(this, target, name='member_div')

#         with tf.Session() as sess:
#             _ = tf.Variable(initial_value='fake_variable')
#             sess.run(tf.global_variables_initializer())

#             main_m = sess.run(_main_norm, {this: feed_dict['this_member']})
#             row_mem = sess.run(_row_norm, {target: feed_dict['target_member']})
#             print(f'main_m: {main_m}')
#             print(f'row_mem: {row_mem}')

#             prod = sess.run(expr_dot, {this: feed_dict['this_member'], target: feed_dict['target_member']})
#             print(f'PROD: {prod}')
#             similarity = sess.run(expr_div, {this: prod, target: (main_m*row_mem)})
#             print(f'SIMILARITY: {similarity}')
            
#             checkpoint_path = os.path.join(self.path, 'recommend_stock_checkpoint', 'similarity.ckpt')
#             saver = tf.train.Saver()
#             saver.save(sess, checkpoint_path, global_step=1000)


# if __name__ == '__main__':
#     rsmodel = RecommendStockModel()
#     rsmodel.hook()
            




# class RecommendSpeedTestService():

#     def __init__(self, member):
#         self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models', 'recommend_stock')
#         self._member = member

#     # def assign(self, member):
#     #     self._member = member

#     def hook(self):
#         sim_dict = self.similarity_calc()
#         recommend_stock_list = self.recommend(sim_dict)
#         print(recommend_stock_list)

#     def similarity_calc(self):
#         m = self._member
#         print(f'MEMBER: \n{m}')

#         members = pd.read_sql_table('members', engine.connect())
#         preprocessing = RecommendStockPreprocessing()
#         refined_members = preprocessing.hook_process(members)
        
#         isExitedMem = refined_members[refined_members['exited']==1].index
#         refined_members = refined_members.drop(isExitedMem)
#         print(f'REFINED MEMBERS: \n{refined_members}')

#         refined_members.set_index(refined_members['email'], inplace=True)
#         refined_members = refined_members.drop(['email'], axis=1)
#         print(f'REFINED MEMBERS AFTER EMAIL INDEXING: \n{refined_members}')

#         base_columns = refined_members.columns

#         this_member = pd.DataFrame(refined_members.loc[m['email'], base_columns]).T
#         else_members = refined_members.loc[:, base_columns].drop(m['email'], axis=0)

#         sim_dict = {}

#         for mem in else_members.index:
#             with tf.Session() as sess:
#                 saver = tf.train.import_meta_graph(os.path.join(self.path, 'recommend_stock_checkpoint', 'cp.ckpt-1000.meta'))
#                 saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.path, 'recommend_stock_checkpoint/')))

#                 graph = tf.get_default_graph()
#                 this = graph.get_tensor_by_name('this_member:0')
#                 target = graph.get_tensor_by_name('target_member:0')
#                 feed_dict = {'this_member': this_member.loc[m['email'], base_columns], 'target_member': else_members.loc[mem, base_columns]}

#                 # _main_norm = graph.get_tensor_by_name('this_norm:0')
#                 # _row_norm = graph.get_tensor_by_name('target_norm:0')
#                 _main_norm = tf.norm(this, name='this_norm')
#                 _row_norm = tf.norm(target, name='target_norm')
#                 expr_dot = graph.get_tensor_by_name('member_dot:0')
#                 expr_div = graph.get_tensor_by_name('member_div:0')

#                 main_m = sess.run(_main_norm, {this: feed_dict['this_member']})
#                 row_mem = sess.run(_row_norm, {target: feed_dict['target_member']})
#                 # print(f'main_m: {main_m}')
#                 # print(f'row_mem: {row_mem}')
#                 prod = sess.run(expr_dot, {this: feed_dict['this_member'], target: feed_dict['target_member']})
#                 # print(f'PROD: {prod}')
#                 similarity = sess.run(expr_div, {this: prod, target: (main_m*row_mem)})

#                 sim_dict[mem] = similarity

#         return sim_dict

#     def recommend(self, sim_dict):
#         m = self._member
#         sim_members = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:50]

#         tradings = pd.read_sql_table('tradings', engine.connect())
#         this_members_tradings = list(tradings[tradings['email'] == m['email']]['stock_ticker'])

#         match_tradings = pd.DataFrame(columns=('id', 'email', 'stock_type', 'stock_ticker', 'stock_qty', 'price', 'trading_date'))
#         for mem, prob in sim_members:
#             match_tradings = pd.concat([match_tradings, tradings[tradings['email'] == mem]])
#         stocks_size = list(match_tradings.groupby('stock_ticker').size().sort_values(ascending=False).index)
#         stocks_list = [{'stock_ticker':s, 
#         'stock_type': str(match_tradings[match_tradings['stock_ticker'] == s]['stock_type'].unique()[0]),
#         'email': m.email} for s in stocks_size if s not in this_members_tradings]
#         return stocks_list

# # if __name__ == '__main__':
# #     member = {'email': '15721779@gmail.com', 'password': '1234', 'name': 'Arnold', 'profile': 'noimage.png', 'geography': 'Spain',
# #     'gender': 'Male', 'age': 41, 'tenure': 5, 'stock_qty': 2, 'balance': 146466, 'has_credit': 0, 'credit_score': 826, 'is_active_member': 0, 
# #     'estimated_salary': 180935, 'role': 'ROLE_USER', 'probability_churn': 0.275768, 'exited': 0}
# #     model = RecommendSpeedTestService(member)
# #     model.hook()

from scipy.spatial.distance import pdist, squareform
        
class RecommendStocksWithSimilarity():

    def hook_process(self, email):
        print('START')
        similarity = self.similarity(email)
        # print(f'get similarity complete')
        sim_members = self.sortFifty(similarity)
        print([mem for mem, sim in sim_members])
        match_tradings = self.similarMembersTradings(sim_members, email)
        print(f'match_tradings: \n{match_tradings}')
        # return pd.DataFrame(match_tradings)
        return [mem for mem, sim in sim_members]

    @staticmethod
    def similarity(email):
        members = pd.read_sql_table('members', engine.connect())
        preprocessing = RecommendStockPreprocessing()
        refined_members = preprocessing.hook_process(members)
        
        isExitedMem = refined_members[refined_members['exited']==1].index
        refined_members = refined_members.drop(isExitedMem)
        isZeroBalMem = refined_members[refined_members['balance']==0].index
        refined_members = refined_members.drop(isZeroBalMem)
        print(len(refined_members))

        refined_members.set_index(refined_members['email'], inplace=True)
        refined_members = refined_members.drop(['email'], axis=1)
        print(f'REFINED MEMBERS: \n{refined_members}')

        base_index = refined_members.index
        base_columns = refined_members.columns

        row_dist = pd.DataFrame(squareform(
            pdist(refined_members, metric='euclidean')), columns=base_index, index=base_index)
        # print(f'ROWDIST: \n {row_dist}')

        this_mem = row_dist[row_dist.index == email]
        this_mem = this_mem.reset_index(drop=True)
        # print(this_mem.to_dict(orient='records'))

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
        return stocks_list


if __name__ == '__main__':
    rs = RecommendStocksWithSimilarity()
    rs.hook_process(email='15660679@gmail.com')





# =====================================================================
# =====================================================================
# ===================        modeling         =========================
# =====================================================================
# =====================================================================



class RecommendStockDto(db.Model):

    __tablename__ = 'recommend_stocks'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    stock_type: str = db.Column(db.String(50), nullable=False)
    stock_ticker: str = db.Column(db.String(100), nullable=False)

    def __init__(self, email, stock_type, stock_ticker):
        self.email = email
        self.stock_type = stock_type
        self.stock_ticker = stock_ticker

    def __repr__(self):
        return f'id={self.id}, email={self.email}, stock_type={self.stock_type}, stock_ticker={self.stock_ticker}'

    @property
    def json(self):
        return {
            'id': self.id,
            'email': self.email,
            'stock_type': self.stock_type,
            'stock_ticker': self.stock_ticker
        }

class RecommendStockVo:
    id: int = 0
    email: str = ''
    stock_type: str =''
    stock_ticker: str = ''







class RecommendStockDao(RecommendStockDto):

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, recommend):
        sql = cls.query.filter(cls.id == recommend.id)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_id(cls, email):
        sql = cls.query.filter(cls.email == email)
        df = pd.read_sql(sql.statement, sql.session.bind)
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(recommend_stock):
        db.session.add(recommend_stock)
        db.session.commit()
    
    @staticmethod
    def modify_recommend_stock(recommend_stock):
        Session = openSession()
        session = Session()
        trading = session.query(RecommendStockDto)\
        .filter(RecommendStockDto.id==recommend_stock.id)\
        .update({RecommendStockDto.stock_type: recommend_stock['stock_type'], RecommendStockDto.stock_id: recommend_stock['stock_id']})
        session.commit()
        session.close()

    @classmethod
    def delete_recommend_stock(cls, id):
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
parser.add_argument('stock_type', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('stock_id', type=str, required=True, help='This field cannot be left blank')

class RecommendStock(Resource):

    @staticmethod
    def post(id):
        body = request.get_json()
        print(f'body: {body}')
        recomm_stock = RecommendStockDto(**body)
        RecommendStockDao.save(recomm_stock)
        return {'recomm_stock': str(recomm_stock.id)}, 200
    
    @staticmethod
    def get(id):
        args = parser.parse_args()
        try:
            recomm_stock = RecommendStockDao.find_by_email(args['email'])
            if recomm_stock:
                return recomm_stock
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404

    @staticmethod
    def put(id):
        args = parser.parse_args()
        print(f'Recommend Stock {args} updated')
        try:
            RecommendStockDao.modify_recommend_stock(args)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404
   
    @staticmethod
    def delete(id):
        try:
            RecommendStockDao.delete_recommend_stock(id)
            return {'code': 0, 'message': 'SUCCESS'}, 200
        except Exception as e:
            print(e)
            return {'message': 'Recommend Stock not found'}, 404
    
class RecommendStocks(Resource):
    
    def post(self):
        rs_dao = RecommendStockDao()
        rs_dao.insert_many('recommend_stocks')

    @staticmethod
    def get(email):
        data = RecommendStockDao.find_by_email(email)
        return data, 200

