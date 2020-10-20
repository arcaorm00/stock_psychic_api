from com_stock_api.ext.db import db 
# from com_stock_api.kospi_pred.dto import KospiDto
# from com_stock_api.korea_covid.dto import KoreaDto
# from com_stock_api.naver_finance.dto import StockDto

class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    news_date : int = db.Column(db.DATETIME, primary_key = True, index=True)
    sentiment_analysis :str = db.Column(db.VARCHAR(30))
    keywords :str = db.Column(db.VARCHAR(30))
    
    def __init__(self,news_date, sentiment_analysis, keywords):
        self.news_date = news_date
        self.sentiment_analysis = sentiment_analysis
        self.keywords = keywords
        
    
    def __repr__(self):
        return f'news_date={self.news_date}, sentiment_analysis={self.sentiment_analysis},\
            keywords={self.keywords}'
            
    @property
    def json(self):
        return {
            'news_date': self.news_date,
            'sentiment_analysis' : self.sentiment_analysis,
            'keywords' : self.keywords
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()