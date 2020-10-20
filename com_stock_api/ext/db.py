from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

db = SQLAlchemy()
Base = declarative_base()

config = {
    'user': 'stockpsychic',
    'password': 'stockpsychic',
    'host': 'stockpsychic.ceq7fgqi0yai.ap-northeast-2.rds.amazonaws.com',
    'port': '3306',
    'database': 'stockpsychic'
}

charset = {'utf8': 'utf8'}

url = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}?charset=utf8'

def openSession():
    ...