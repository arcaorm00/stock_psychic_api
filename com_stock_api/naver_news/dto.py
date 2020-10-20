from com_stock_api.ext.db import db 
# from com_stock_api.kospi_pred.dto import KospiDto
# from com_stock_api.korea_covid.dto import KoreaDto
# from com_stock_api.naver_finance.dto import StockDto

class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : int = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    neg : float = db.Column(db.Float)
    pos : float = db.Column(db.Float)
    neu : float = db.Column(db.Float)
    keywords :str = db.Column(db.VARCHAR(30))
    url :str = db.Column(db.VARCHAR(30))

    
    def __init__(self, id, date, headline, neg, pos, neu, keywords,url):
        self.id = id
        self.date = date
        self.headline = headline
        self.neg = neg
        self.pos = pos
        self.neu = neu
        self.keywords = keywords
        self.url = url
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            neg={self.neg},pos={self.pos},neu={self.neu},keywords={self.keywords},url={self.url}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'neg':self.neg,
            'pos':self.pos,
            'neu':self.neu,
            'keywords':self.keywords,
            'url':self.url
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()