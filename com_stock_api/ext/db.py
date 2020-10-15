from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': '3306',
    'database': 'stockdb'
}