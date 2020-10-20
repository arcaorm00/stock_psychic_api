import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.request import Request
# from nltk.sentiment.vader import SentimentIntensityAnalyzer



class YHNewsScrap:

    ticker : str
    finviz_url = 'https://finviz.com/quote.ashx?t='

    def __init__(self):
        self.ticker = ''


    def __init__(self, ticker):
        self.ticker = ticker

    
    def hook(self):
        self.get_latest_news()

    def get_latest_news(self):
        ticker = self.ticker
        url = self.finviz_url

    # Open url
        url +=ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
        resp = urlopen(req)
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')

    # Extract news data only

        print(type(news_table))

        df = news_table.findAll('tr')



        print('\n')
        print ('Recent News Headlines for {}: '.format(ticker))

        n = 5

        month = ['Jan', 'Fab', 'Mar', 'Apr', 'May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for i, row in enumerate(df):
            headline = row.a.text
            time = row.td.text
            time = time.strip()
            
            for j in month:
                if time[:3] == j:
                    date = time[:9]



            link = row.find("a").get("href")
            print(headline,'(', date , ')', 'link:', link)
            if i == n-1:
                break
         



if __name__=='__main__':
    news_pro = YHNewsScrap('AAPL')
    news_pro.hook()