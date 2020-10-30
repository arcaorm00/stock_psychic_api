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

    #@staticmethod
    def bulk(self):
        #service = CovidService()
        #df = service.hool()
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
    def fond_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_id(cls,id):
        return cls.query.filter_by(id == id).all()


    @classmethod
    def find_by_date(cls, date):
        return cls.query.filter_by(date == date).first()

    @classmethod
    def login(cls,covid):
        sql = cls.query.fillter(cls.id.like(covid.id)).fillter(cls.date.like(covid.date))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('======================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

if __name__ == '__main__':
    #KoreaDao.bulk()
    k = KoreaDao()
    k.bulk()    


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('seoul_cases',type=int, required=True,help='This field should be a seoul_cases')
parser.add_argument('seoul_death',type=int, required=True,help='This field should be a seoul_deate')
parser.add_argument('total_cases',type=int, required=True,help='This field should be a total_cases')
parser.add_argument('total_death',type=int, required=True,help='This field should be a password')


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
    def post(id):
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
    def get():
        kd = KoreaDao()
        kd.insert_many('korea_covid')
    
    @staticmethod
    def get():
        print('======kc========')
        data = KoreaDao.find_all()
        return data, 200

class Auth(Resource):

    @staticmethod
    def post():
        body = request.get_json()
        covid = KoreaDto(**body)
        KoreaDao.save(covid)
        id = covid.id

        return {'id': str(id)}, 200

class Access(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        covid = KoreaVo()
        covid.id = args.id
        covid.date = args.date
        print(covid.id)
        print(covid.date)
        data = KoreaDao.login(covid)
        return data[0], 200
