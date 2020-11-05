import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json


# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================

class Covidedit():
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    
    def csv(self):
        path = self.data
        df_kor = pd.read_csv(path +'/kr_daily.csv')
        df_reg = pd.read_csv(path +'/kr_regional_daily.csv')
        del df_kor['released']
        del df_kor['tested']
        del df_kor['negative']
        df_kor.columns =['date','total_cases','total_death']
        df_kor['date']=pd.to_datetime(df_kor['date'].astype(str), format='%Y/%m/%d')
        print(df_kor)

        #print(df_reg)
        df_reg = df_reg[df_reg['region']=='서울']
        #print(df_reg)
        del df_reg['region']
        del df_reg['released']
        df_reg.columns =['date','seoul_cases','seoul_death']
        #print(df_reg)
        df_reg['date']=pd.to_datetime(df_reg['date'].astype(str), format='%Y/%m/%d')
        print(df_reg)
        
        df_all = pd.merge(df_kor,df_reg, on=['date','date'],how='left')
        df_all = df_all.fillna(0)
        df_all['seoul_cases'] = df_all['seoul_cases'].astype(int)
        df_all['seoul_death'] = df_all['seoul_death'].astype(int)
        df_all.set_index('date', inplace=True)
        print(df_all)

        df_all.to_csv(path + '/kor&seoul.csv',encoding='UTF-8')

if __name__ == '__main__':
    #Covidedit()
    c=Covidedit()
    c.csv()

class KoreaDto(db.Model):
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    seoul_cases : int = db.Column(db.String(30))
    seoul_death : int = db.Column(db.String(30))
    total_cases : int = db.Column(db.String(30))
    total_death : int = db.Column(db.String(30))
    
    def __init__(self, id,date, seoul_cases, seoul_death, total_cases, total_death):
        self.date = date
        self.seoul_cases = seoul_cases
        self.seoul_death = seoul_death
        self.total_cases = total_cases
        self.total_death = total_death
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, seoul_cases={self.seoul_cases},\
            seoul_death={self.seoul_death},total_cases={self.total_cases},total_deatb={self.total_death}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'seoul_cases' : self.seoul_cases,
            'seoul_death' : self.seoul_death,
            'total_cases' : self.total_cases,
            'total_death' : self.total_death
        }

class KoreaVo:
    id : int = 0
    date: str = ''
    seoul_cases : int =''
    seoul_death : int =''
    total_cases : int =''
    total_deat : int =''

Session = openSession()
session= Session()



class KoreaDao(KoreaDto):
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")


    def bulk(self):
        path = self.data
        df = pd.read_csv(path +'/kor&seoul.csv', encoding='utf-8')
        print(df.head())
        session.bulk_insert_mappings(KoreaDto, df.to_dict(orient='records'))
        session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(KoreaDto.id)).one()
    
    @staticmethod
    def save(covid):
        db.session.add(covid)
        db.session.commit()

    @staticmethod
    def update(covid):
        db.session.add(covid)
        db.session.commit()
    
    @classmethod
    def delete(cls,date):
        data = cls.qeury.get(date)
        db.session.delete(data)
        db.sessio.commit()

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_id(cls,id):
        return session.query(KoreaDto).filter(KoreaDto.id.like(id)).one()
    @classmethod
    def find_by_seoulcases(cls,seoul_cases):
        return session.qeury(KoreaDto).filter(KoreaDto.seoul_cases.like(seoul_cases)).one()

    @classmethod
    def find_by_seouldeath(cls,seoul_death):
        return session.query(KoreaDto).filter(KoreaDto.seoul_death.like(seoul_death)).one()

    @classmethod
    def find_by_totalcases(cls,total_cases):
        return session.query(KoreaDto).filter(KoreaDto.total_cases.like(total_cases)).one()
    
    @classmethod
    def find_by_totaldeath(cls,total_death):
        return session.query(KoreaDto).filter(KoreaDto.total_death.like(total_death)).one()
    
    @classmethod
    def find_by_date(cls, date):
        return session.query.filter(KoreaDto.date.like(date)).one()



# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('date',type=str, required=True,help='This field cannot be left blank')
parser.add_argument('seoul_cases',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('seoul_death',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('total_cases',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('total_death',type=int, required=True,help='This field cannot be left blank')


class KoreaCovid(Resource):
    
    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Covid {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'
        params_str = ''
        for key in params.keys():
            params_str += 'key: {}, value:{}<br>'.format(key, params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
        
    
    @staticmethod
    def get(id: int):
        print(f'Covid {id} added')
        try:
            covid = KoreaDao.find_by_id(id)
            if covid:
                return covid.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'Covid {args["id"]} updated')
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Covid {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class KoreaCovids(Resource):

    @staticmethod
    def post():
        kd = KoreaDao()
        kd.insert_many('korea_covid')
    
    @staticmethod
    def get():
        print('======kc========')
        data = KoreaDao.find_all()
        return data, 200


