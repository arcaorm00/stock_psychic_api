from com_stock_api.ext.db import db 
# from com_stock_api.kospi_pred.dto import KospiDto
# from com_stock_api.naver_finance.dto import StockDto
# from com_stock_api.naver_news.dto import NewsDto

class KoreaDto(db.Model):
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : int = db.Column(db.DATETIME)
    seoul_cases :int = db.Column(db.VARCHAR(30))
    seoul_death :int = db.Column(db.VARCHAR(30))
    total_cases : int = db.Column(db.VARCHAR(30))
    total_death : int = db.Column(db.VARCHAR(30))
    
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

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()




  