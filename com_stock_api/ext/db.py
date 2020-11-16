from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db = SQLAlchemy()
Base = declarative_base()

# config = {
#     'user': 'stockpsychic',
#     'password': 'stockpsychic',
#     'host': 'stockpsychic.ceq7fgqi0yai.ap-northeast-2.rds.amazonaws.com',
#     'port': '3306',
#     'database': 'stockpsychic'
# }


config = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'port': '3306',
    'database': 'stockdb'
}

charset = {'utf8': 'utf8'}

url = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}?charset=utf8'
Base = declarative_base()
engine = create_engine(url, pool_size=30, max_overflow=0)

def openSession():
    return sessionmaker(bind=engine)