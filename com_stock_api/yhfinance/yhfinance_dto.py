from com_stock_api.ext.db import db

class YHFinanceDto(db.Model):
    __tablename__ = 'Yahoo_Finance'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str  = db.Column(db.Date)
    open : float = db.Column(db.Float)
    close : float = db.Column(db.Float)
    adjclose : float = db.Column(db.Float)
    high : float = db.Column(db.Float)
    low : float = db.Column(db.Float)
    amount : int = db.Column(db.Integer)

    #date format : YYYY-MM-DD
    # amount : unit = million 
    
    def __repr__(self):
        return f'YHFinance(id=\'{self.id}\', date=\'{self.date}\',open=\'{self.open}\', \
            close=\'{self.close}\',adjclose=\'{self.adjclose}\',high=\'{self.high}\',\
            low=\'{self.low}\',amount=\'{self.amount}\')'


    @property
    def json(self):
        return {
            'id' : self.id,
            'date' : self.date,
            'open' : self.open,
            'close' : self.close,
            'adjclose' : self.adjclose,
            'high' : self.high,
            'low' : self.low,
            'amount' : self.amount,
            'date' : self.date
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()