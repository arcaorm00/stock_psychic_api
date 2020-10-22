from com_stock_api.ext.db import db 
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from com_stock_api.naver_news.analysis import NewsAnalysis
# from com_stock_api.korea_covid.dto import KoreaDto
# from com_stock_api.kospi_pred.dto import KospiDto
# from com_stock_api.naver_finance.dto import StockDto

config = {
    'user' : 'root',
    'password' : 'root',
    'host': '127.0.0.1',
    'port' : '3306',
    'database' : 'stockdb'
}
charset ={'utf8':'utf8'}
url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"

engine = create_engine(url)

class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    contents : str = db.Column(db.String(10000))
    url :str = db.Column(db.String(500))
    stock : str = db.Column(db.String(30))
    label : float = db.Column(db.Float)



    #'date', 'headline', 'contents', 'url', 'stock', 'label'

    
    def __init__(self, id, date, headline, contents, url, stock, label):
        self.id = id
        self.date = date
        self.headline = headline
        self.contents = contents
        self.url = url
        self.stock = stock
        self.label = label
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            contents={self.contents},url={self.url},stock={self.stock},label={self.label}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'contents':self.contents,
            'url':self.url,
            'stock':self.stock,
            'label':self.label
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

# service = NewsAnalysis()
# Session = sessionmaker(bind = engine)
# s = Session()
# df = service.makelabel()
# print(df.head())
# s.bulk_insert_mappings(NewsDto, df.to_dict(orient="records"))
# s.commit()
# s.close()