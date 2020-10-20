from com_stock_api.ext.db import db
from com_stock_api.member.member_dto import MemberDto
from com_stock_api.yhfinance.yhfinance_dto import YHFinanceDto
from com_stock_api.naver_finance.dto import StockDto
import datetime

class TradingDto(db.Model):

    __tablename__ = "tradings"
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email: str = db.Column(db.String(100), db.ForeignKey(MemberDto.email), nullable=False)
    kospi_stock_id: int = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    nasdaq_stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    stock_qty: int = db.Column(db.Integer, nullable=False)
    price: float = db.Column(db.Integer, nullable=False)
    trading_date: str = db.Column(db.String(1000), default=datetime.datetime.now())

    def __init__(self, id, email, kospi_stock_id, nasdaq_stock_id, stock_qty, price, trading_date):
        self.id = id
        self.email = email
        self.kospi_stock_id = kospi_stock_id
        self.nasdaq_stock_id = nasdaq_stock_id
        self.stock_qty = stock_qty
        self.price = price
        self.trading_date = trading_date

    def __repr__(self):
        return 'Trading(trading_id={}, member_id={}, kospi_stock_id={}, nasdaq_stock_id={}, stock_qty={}, price={}, trading_date={})'.format(self.id, self.member_id, self.kospi_stock_id, self.nasdaq_stock_id, self.stock_qty, self.price, self.trading_date)
    
    @property
    def json(self):
        return {
            'id': self.id,
            'member_id': self.member_id,
            'kospi_stock_id': self.kospi_stock_id,
            'nasdaq_stock_id': self.nasdaq_stock_id,
            'stock_qty': self.stock_qty,
            'price': self.price,
            'trading_date': self.trading_date
        }

    def save(self):
        db.session.add(self)
        db.commit()

    def delete(self):
        db.session.delete(self)
        db.commit()