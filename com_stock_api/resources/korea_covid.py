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
        df_kor.columns =['date','total_cases','total_deaths']
        df_kor['date']=pd.to_datetime(df_kor['date'].astype(str), format='%Y/%m/%d')
        print(df_kor)

        #print(df_reg)
        df_reg = df_reg[df_reg['region']=='서울']
        #print(df_reg)
        del df_reg['region']
        del df_reg['released']
        df_reg.columns =['date','seoul_cases','seoul_deaths']
        #print(df_reg)
        df_reg['date']=pd.to_datetime(df_reg['date'].astype(str), format='%Y/%m/%d')
        print(df_reg)
        
        df_all = pd.merge(df_kor,df_reg, on=['date','date'],how='left')
        df_all = df_all.fillna(0)
        df_all['seoul_cases'] = df_all['seoul_cases'].astype(int)
        df_all['seoul_deaths'] = df_all['seoul_deaths'].astype(int)
        df_all.set_index('date', inplace=True)
        print(df_all)

        df_all.to_csv(path + '/kor&seoul.csv',encoding='UTF-8')

# if __name__ == '__main__':
#     #Covidedit()
#     c=Covidedit()
#     c.csv()

class KoreaDto(db.Model):
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    seoul_cases : int = db.Column(db.String(30))
    seoul_deaths : int = db.Column(db.String(30))
    total_cases : int = db.Column(db.String(30))
    total_deaths : int = db.Column(db.String(30))
    
    def __init__(self, id,date, seoul_cases, seoul_deaths, total_cases, total_deaths):
        self.date = date
        self.seoul_cases = seoul_cases
        self.seoul_death = seoul_death
        self.total_cases = total_cases
        self.total_death = total_death
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, seoul_cases={self.seoul_cases},\
            seoul_death={self.seoul_deaths},total_cases={self.total_cases},total_deaths={self.total_deaths}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'seoul_cases' : self.seoul_cases,
            'seoul_deaths' : self.seoul_deaths,
            'total_cases' : self.total_cases,
            'total_deaths' : self.total_deaths
        }

class KoreaVo:
    id : int = 0
    date: str = ''
    seoul_cases : int =''
    seoul_deaths : int =''
    total_cases : int =''
    total_deaths : int =''

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
    def find_by_seouldeaths(cls,seoul_death):
        return session.query(KoreaDto).filter(KoreaDto.seoul_death.like(seoul_death)).one()

    @classmethod
    def find_by_totalcases(cls,total_cases):
        return session.query(KoreaDto).filter(KoreaDto.total_cases.like(total_cases)).one()
    
    @classmethod
    def find_by_totaldeaths(cls,total_death):
        return session.query(KoreaDto).filter(KoreaDto.total_death.like(total_death)).one()
    
    @classmethod
    def find_by_date(cls, date):
        return session.query(KoreaDto).filter(KoreaDto.date.like(date)).all()



# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('date',type=str, required=True,help='This field cannot be left blank')
parser.add_argument('seoul_cases',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('seoul_deaths',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('total_cases',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('total_deaths',type=int, required=True,help='This field cannot be left blank')


class KoreaCovid(Resource):
    
    @staticmethod
    def post(self):
        data = self.parser.parse_args()
        kcovid = KoreaDto(data['date'],data['seoul_cases'],data['seoul_deaths'],data['total_cases'],data['total_deaths'])
        try:
            kcovid.save(data)
            return {'code':0, 'message':'SUCCESS'},200
        except:
            return {'message': 'An error occured inserting recent news'}, 500
        return kcovid.json(), 201
        
    
    def get(self):
        kcovid = KoreaDao.find_all()
        return kcovid , 200

    
    def put(self, id):
        data = KoreaCovid.parser.parse_args()
        
        kcovid = KoreaDao.find_by_id(id)
        kcovid.date = data['date']
        kcovid.total_cases = data['total_cases']
        kcovid.total_deaths = data['total_deaths']
        kcovid.seodul_cases = data['seoul_cases']
        kcovid.seoul_deaths = data['seoul_deaths']
        kcovid.save()
        return kcovid.json()

class KoreaCovids(Resource):
    def get(self):
        return KoreaDao.find_all(), 200


