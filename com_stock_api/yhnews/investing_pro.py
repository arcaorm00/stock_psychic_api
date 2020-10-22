from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import requests
import pandas as pd
from unicodedata import normalize
from datetime import datetime, date
import re
import os
from http.client import IncompleteRead
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class InverstingPro:

    tickers = {'AAPL' : 'apple-computer-inc', 'TSLA' : 'tesla-motors'}
    ticker_str : str
    ticker : str
    n : int
    # Temporaily set up the page number for the time sake
    AAPL = 240
    TSLA = 150
    START_DATE = datetime.strptime('2020-01-01', '%Y-%m-%d')
    END_DATE = datetime.strptime('2020-06-30', '%Y-%m-%d')
  
    def __init__(self):
        ticker = input(print("Please enter a stock symbol: "))
        if (self.tickers[ticker]) != None:
            self.ticker_str = self.tickers.get(ticker)
            self.ticker = ticker
            self.n = self.AAPL if self.ticker == 'AAPL' else self.TSLA
            self.processed_info = []

        else:
            print(KeyError)

    def __init__(self, ticker):
        if (self.tickers[ticker]) != None:
            self.ticker_str = self.tickers.get(ticker)
            self.ticker = ticker
            self.n = self.AAPL if self.ticker == 'AAPL' else self.TSLA
            self.processed_info = []
        else:
            print(KeyError)

    def hook(self):

        news_pages  =[]
        while(self.n>1):
            news_page = self.get_stock_news_pages(self.ticker_str, self.n)
            # Get all the pages within the date range (Possible to include other dates \
            # in first and last news page due to multiple articles on one page)
            if (self.dates_checker(news_page)):
                print("Getting ",self.n,"th news page...")
                news_pages.append(news_page)
            else:
                #Device to take START_DATE pages and to stop bringing pages when it reaches to the END_DATE
                if (len(news_pages) > 0):
                    break
            self.n-=1

        article_pages=[]
        i = 1
        for index, p in enumerate(news_pages):
            links = self.get_internal_article_links(p)
            print('Processing ... {} / {}'.format(i,len(news_pages) ))

            for l in reversed(links):
                page = self.get_article_page(l)
                if (index ==0 or index == len(news_page) -1):
                    t= datetime.strptime(self.get_publish_time(page), '%Y-%m-%d')
                    if (t > self.END_DATE or t < self.START_DATE):
                        continue
                
                text = self.extract_text(page)
                headline = self.get_headline(page)
                date = self.get_publish_time(page)
                self.processed_info.append([l, date, headline, text])
     
            i+=1

        # Add a ticker in the list 
        for i in self.processed_info:
            i.insert(1, self.ticker)

        self.save_news(self.processed_info)
        self.get_sentiment_analysis(self.processed_info)

        print("++ DONE ++")

        

    def save_news(self, news_list):
        col = ['Date', 'Ticker', 'Link', 'Headline', 'Content']
        df = pd.DataFrame(news_list, columns=col)
        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = self.ticker + '_news.csv'
        output_file = os.path.join(path,file_name)
        df.to_csv(output_file)
        print("Completed saving ", file_name)


    def get_stock_news_pages(self, stock_string, n):
        # pages =[]
        request = Request('https://www.investing.com/equities/' + stock_string + '-news/' + str(n), headers={"User-Agent": "Mozilla/5.0"})
        content = urlopen(request).read()
        # pages.append(bs(content, 'html.parser'))

        return bs(content, 'html.parser')

    def dates_checker (self, page):

        tags = page.find_all('span', attrs={'class' : 'date'})[:10]
        dates = [i.text.strip().replace("-", '').replace('\xa0','') for i in tags]
        published_dates=[]
        for i in dates:
            try:
                published_dates.append(datetime.strptime(i, '%b %d, %Y'))
            except ValueError:
                pass

        published_dates[:] = [date for date in published_dates if (self.START_DATE <= date <= self.END_DATE)]
        return False if (len(published_dates) == 0) else True
    
    def date_checker (self, article):
        ...
        

    def get_headline(self, page):
        headline = page.find('title').text
        return headline

    def get_internal_article_links(self, page):
        news = page.find_all('div', attrs={'class': 'mediumTitle1'})[1]
        articles = news.find_all('article', attrs={'class': 'js-article-item articleItem'})
        return ['https://www.investing.com' + a.find('a').attrs['href'] for a in articles]

    def get_article_page(self, article_link):
        request = Request(article_link, headers={"User-Agent": "Mozilla/5.0"})
        content = urlopen(request).read()
        
        try:
            page= bs(content, 'lxml')
        except IncompleteRead as e:
            print("Incomplete Read error occurs")
            page = e.partial
        return page

    def clean_paragraph(self, paragraph):
        paragraph = re.sub(r'\(http\S+', '', paragraph)
        paragraph = re.sub(r'\([A-Z]+:[A-Z]+\)', '', paragraph)
        paragraph = re.sub(r'[\n\t\s\']', ' ', paragraph)
        return normalize('NFKD', paragraph)    

    def extract_text(self, article_page):
        text_tag = article_page.find('div', attrs={'class': 'WYSIWYG articlePage'})
        paragraphs = text_tag.find_all('p')
        text = '\n'.join([self.clean_paragraph(p.get_text()) for p in paragraphs[:-1]])
        text = "".join(text)
        return text

    def get_publish_time(self, article_page):
        details = article_page.find('meta', attrs={'itemprop': 'dateModified'})
        publish_date = details.get_attribute_list('content')[0]
        publish_date = str(datetime.strptime(publish_date, '%Y-%m-%d %H:%M:%S'))
        publish_date = "".join(publish_date)
        publish_date = publish_date[:10]
        return publish_date

    def get_sentiment_analysis(self, news_list):

        vader = SentimentIntensityAnalyzer()
        col = ['Date', 'Ticker', 'Link', 'Headline', 'Content']
        news_with_scores = pd.DataFrame(news_list, columns=col)
        scores = news_with_scores['Content'].apply(vader.polarity_scores).tolist()
        
        scores_df = pd.DataFrame(scores)
        news_with_scores = news_with_scores.join(scores_df, rsuffix='_right')
        news_with_scores['Date'] = pd.to_datetime(news_with_scores.Date).dt.date
        news_with_scores.drop('Content',axis=1,inplace=True)

        path = os.path.abspath(__file__+"/.."+"/data/")
        file_name = self.ticker + '_sentiment.csv'
        output_file = os.path.join(path,file_name)
        news_with_scores.to_csv(output_file)
        print("Completed saving ", file_name)

        print(news_with_scores.head())

if __name__=='__main__':
    tesla = InverstingPro('TSLA')
    apple = InverstingPro('AAPL')
    tesla.hook()
    apple.hook()

