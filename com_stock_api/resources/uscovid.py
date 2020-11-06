from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from sqlalchemy.orm import Session, sessionmaker, joinedload
from sqlalchemy import create_engine
import pandas as pd
import os
from sqlalchemy import and_,or_,func
import json


# =============================================================
# =============================================================
# ======================      SERVICE    ======================
# =============================================================
# =============================================================

class USCovidPro : 
    def __init__(self):
        path = os.path.abspath(__file__+"/.."+"/data")

# 1. Preprocessing California covid cases which is organized by county to the total numebers
# '''
# df = pd.read_csv(path + "/statewide_cases.csv")

# df.drop('county', axis = 1, inplace=True)
# df.drop('newcases', axis = 1, inplace=True)
# df.drop('newdeaths', axis = 1, inplace=True)
# # print(df.head())
  
# dfout = df.groupby(['date']).sum()
# dfout.reset_index(level=0, inplace=True)
# finaldf = dfout[['date', 'cases', 'deaths']]
# finaldf.columns = ['date', 'ca_cases', 'ca_deaths']
# finaldf.to_csv(path+"/ca.csv", index=False)


# # print(finaldf.head())
# '''

#2. Change their date type, float to integer
    def hook(self):
        df = pd.read_csv(self.path+"/ca.csv")
        df_us = pd.read_csv(self.path+"/us.csv")

        df_all = pd.merge(df_us, df, on=['date','date'], how='left')

        df_all['total_cases'] = pd.to_numeric(df_all['total_cases'], errors='coerce').fillna(0).astype(int)
        df_all['total_deaths'] = pd.to_numeric(df_all['total_deaths'], errors='coerce').fillna(0).astype(int)
        df_all['ca_cases'] = pd.to_numeric(df_all['ca_cases'], errors='coerce').fillna(0).astype(int)
        df_all['ca_deaths'] = pd.to_numeric(df_all['ca_deaths'], errors='coerce').fillna(0).astype(int)

        df_all.to_csv(self.path+"/covid.csv")
        print(df_all.head())


# =============================================================
# =============================================================
# =====================    MODELING      ======================
# =============================================================
# =============================================================

class USCovidDto(db.Model):
    __tablename__ = 'US_Covid_cases'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date: str = db.Column(db.Date)
    total_cases: int = db.Column(db.Integer)
    total_deaths: int = db.Column(db.Integer)
    ca_cases : int = db.Column(db.Integer)
    ca_deaths: int = db.Column(db.Integer)
    #date format : YYYY-MM-DD
    
    def __init__(self, date, total_cases, total_deaths, ca_cases, ca_deaths):
        self.date = date
        self.total_cases = total_cases
        self.total_deaths = total_deaths
        self.ca_cases = ca_cases
        self.ca_deaths = ca_deaths

    def __repr__(self):
        return f'USCovid(id=\'{self.id}\', date=\'{self.date}\', total_cases=\'{self.total_cases}\',\
            total_deaths=\'{self.total_deaths}\',ca_cases=\'{self.ca_cases}\', \
                ca_deaths=\'{self.ca_deaths}\')'


    @property
    def json(self):
        return {
            'id' : self.id,
            'date' : self.date,
            'total_cases' : self.total_cases,
            'total_deaths' : self.total_death,
            'ca_cases' : self.ca_cases,
            'ca_deaths' : self.ca_death,
        }

class USCovidVo:
    id: int = 0
    date: str = ''
    total_cases: int = 0
    total_deaths: int = 0
    ca_cases: int = 0
    ca_deaths: int = 0

Session = openSession()
session = Session()

class USCovidDao(USCovidDto):
    @staticmethod
    def count():
        return session.query(func.count(USCovidDto.id)).one()
    @staticmethod
    def save(data):
        db.session.add(data)
        db.session.commit()
    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()
    @staticmethod
    def delete(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
    @staticmethod
    def bulk():
        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = 'covid.csv'
        input_file = os.path.join(path,file_name)
        df = pd.read_csv(input_file)
        print(df.head())
        session.bulk_insert_mappings(USCovidDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @classmethod
    def find_by_date(cls, date):
        return session.query(USCovidDto).filter(USCovidDto.date.like(date)).all()
    @classmethod
    def find_by_period(cls,start_date, end_date):
        return session.query(USCovidDto).filter(date__range=(start_date, end_date)).all()
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_only_us(cls):
        return session.query(USCovidDto).with_entities(USCovidDto.total_cases, USCovidDto.total_deaths)

    @classmethod
    def find_only_ca(cls):
        return session.query(USCovidDto).with_entities(USCovidDto.ca_cases, USCovidDto.ca_deaths)

# =============================================================
# =============================================================
# =====================    CONTROLLER    ======================
# =============================================================
# =============================================================

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('total_cases', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('total_deaths', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('ca_cases', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('ca_deaths', type=int, required=False, help='This field cannot be left blank')

class USCovid(Resource):       
    def post(self):
        data = self.parset.parse_args()
        uscovid = USCovidDto(data['date'], data['total_cases'], data['total_deaths'], data['ca_cases'], data['ca_deaths'])
        try: 
            uscovid.save(data)
            return {'code': 0, 'message' : 'SUCCESS'}, 200
        except:
            return {'message': 'An error occured inserting the covid case'}, 500
        return uscovid.json(), 201
    def get(self):
        print("=====uscovid.py / uscovid's get")
        data = USCovidDao.find_all()
        return data, 200
    @staticmethod
    def fetch(date:str):
        print("=====uscovid.py / uscovid's fetch")
        uscovid = USCovidDao.find_by_id(date)
        if uscovid:
            return uscovid.json()
        return {'message': 'uscovid not found'}, 404
    def put(self, id):
        data = USCovid.parser.parse_args()
        uscovid = USCovidDao.find_by_id(id)
        uscovid.date = data['date']
        uscovid.total_cases = data['total_cases']
        uscovid.total_deaths = data['total_deaths']
        uscovid.ca_cases = data['ca_cases']
        uscovid.ca_deaths = data['ca_deaths']
        uscovid.save()
        return uscovid.json()

class USNewCases(Resource):
    @staticmethod
    def get():
        print("====uscovid.py / TotalCases's get ")
        query = USCovidDao.find_only_us()
        df = pd.read_sql_query(query.statement, query.session.bind)
        df['new_cases'] = df.total_cases.diff().fillna(0)
        df['new_death'] = df.total_deaths.diff().fillna(0)
        df =df.astype(int)
        data = json.loads(df.to_json(orient="records"))
        return data, 200



class CANewCases(Resource):
    @staticmethod
    def get():
        print("====uscovid.py / TotalCases's get ")
        query = USCovidDao.find_only_ca()
        df = pd.read_sql_query(query.statement, query.session.bind)
        df['new_cases'] = df.ca_cases.diff().fillna(0)
        df['new_death'] = df.ca_deaths.diff().fillna(0)
        df =df.astype(int)
        data = json.loads(df.to_json(orient="records"))
        return data, 200

class USCovids(Resource):
    def get():
        return USCovidDao.find_all(), 200

if __name__ == "__main__":
    CANewCases.get()