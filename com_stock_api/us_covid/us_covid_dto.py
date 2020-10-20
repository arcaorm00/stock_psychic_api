from com_stock_api.ext.db import db

class USCovidDto(db.Model):
    __tablename__ = 'US_Covid_cases'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date: str = db.Column(db.Date)
    total_cases: int = db.Column(db.Integer)
    total_death: int = db.Column(db.Integer)
    ca_cases : int = db.Column(db.Integer)
    ca_death: int = db.Column(db.Integer)
    #date format : YYYY-MM-DD
    
    def __repr__(self):
        return f'USCovid(id=\'{self.id}\', date=\'{self.date}\', total_cases=\'{self.total_cases}\',\
            total_death=\'{self.total_death}\',ca_cases=\'{self.ca_cases}\', \
                ca_death=\'{self.ca_death}\')'


    @property
    def json(self):
        return {
            'id' : self.id,
            'date' : self.date,
            'total_cases' : self.total_cases,
            'total_death' : self.total_death,
            'ca_cases' : self.ca_cases,
            'ca_death' : self.ca_death,
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()