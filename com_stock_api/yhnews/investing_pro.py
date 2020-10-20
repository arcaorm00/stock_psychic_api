from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
import requests
import pandas as pd
# import v3io_frames as v3f
from unicodedata import normalize
from datetime import datetime, date
import re
import os
import json
# from nltk.sentiment.vader import SentimentIntensityAnalyzer


class InverstingPro:

    tickers = {'AAPL' : 'apple-computer-inc', 'TSLA' : 'tesla-motors'}
    ticker_str : str
    ticker : str
    n : int
    AAPL = 240
    TSLA = 150
    def __init__(self):
        ...

    def __init__(self, ticker):
        if (self.tickers[ticker]) != None:
            self.ticker_str = self.tickers.get(ticker)
            self.ticker = ticker
            self.n = self.num = self.AAPL if self.ticker == 'AAPL' else self.TSLA
            self.processed_info = []
        else:
            print(KeyError)

    def hook(self):

        news_pages  =[]
        news_page = self.get_stock_news_pages(self.ticker_str, self.n)
        if (self.dates_checker(news_page) and abs(self.num-self.n) <=100): #Only want to see 100 pages for 6 months long
            news_pages.append(news_page)
            n-=1

        article_pages=[]
        for p in news_pages:
            links = self.get_internal_article_links(p)
            for l in links:
                page = self.get_article_page(l)
                text = self.extract_text(page)
                headline = self.get_headline(page)
                date = self.get_publish_time(page)
                self.processed_info.append([l, date, headline, text])
        
        print(self.processed_info)


        # date checker return value : 0 = All the articles are out of range.
        #                           : 1 = This page is good to read all (ex. 2020-10-30)


        # links = self.get_internal_article_links(news_pages)
        # for link in links:
        #     page = self.get_article_page(link)
        #     text = self.extract_text(page)
        #     headline = self.get_headline(page)
        #     date = self.get_publish_time(page)
        #     info = [self.ticker, date, headline, text, link]
        #     self.processed_info.append(info)
        # return self.processed_info
        


    def get_stock_news_pages(self, stock_string, n):
        # pages =[]
        request = Request('https://www.investing.com/equities/' + stock_string + '-news/' + str(n), headers={"User-Agent": "Mozilla/5.0"})
        content = urlopen(request).read()
        # pages.append(bs(content, 'html.parser'))

        return bs(content, 'html.parser')

    def dates_checker (self, page):

        tags = page.find_all('span', attrs={'class' : 'date'})[:10]
        published_dates = [i.text.strip().replace("-", '').replace('\xa0','') for i in tags]
        published_dates = [datetime.strptime(i, '%b %d, %Y') for i in published_dates]
        # strftime('%Y-%m-%d')


        for t in published_dates:
            if (datetime(2020,1,1) <= t <= datetime(2020, 6, 30)):
                continue
            else:
                published_dates.remove(t)
            
        
        return False if len(published_dates) == 0 else True
        
        

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
        return bs(content, 'html.parser')

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

    def get_score(self, paragraph_scores):
        return sum([score - 1 for score in paragraph_scores]) / len(paragraph_scores)  

    def get_article_scores(self, context, articles, endpoint):
        scores = [] 
        for i, article in enumerate(articles):
            context.logger.info(f'getting score for article {i + 1}\\{len(articles)}')
            event_data = {'instances': article.split('\n')}
            resp = requests.put(endpoint+'/bert_classifier_v1/predict', json=json.dumps(event_data))
            scores.append(get_score(json.loads(resp.text)))
        return scores


investing = InverstingPro('TSLA')


news_page = investing.get_stock_news_pages(investing.ticker_str, investing.n)
a = investing.dates_checker(news_page)

# for i in range(len(links)):
#     print("Time: ", time[i], " pages: " , pages[i], " links: ", links[i])