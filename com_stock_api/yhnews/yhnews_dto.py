from com_stock_api.ext.db import db

class YHNewsDto(db.Model):
    __tablename__ = 'Yahoo_News'
    __table_args__={'mysql_collate':'utf8_general_ci'}
        # , primary_key = True, index = True

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.Date)
    ticker : str = db.Column(db.String(30)) #stock symbol
    headline : str = db.Column(db.String(255))
    neg : float = db.Column(db.Float)
    pos : float = db.Column(db.Float)
    neu : float = db.Column(db.Float)
    compound :float  = db.Column(db.Float)



    def __repr__(self):
        return f'User(id=\'{self.id}\', date=\'{self.date}\',ticker=\'{self.ticker}\',\
                headline=\'{self.headline}\',neg=\'{self.neg}\', \
                pos=\'{self.pos}\',neu=\'{self.neu}\', \
                compound=\'{self.compound}\',)'


    @property
    def json(self):
        return {
            'id': self.id,
            'date' : self.date,
            'ticker' : self.ticker,
            'headline' : self.headline,
            'neg' : self.neg,
            'pos' : self.pos,
            'neu' : self.neu,
            'compound' : self.compound
        }


    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()
