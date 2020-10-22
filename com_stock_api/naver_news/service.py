# -*- coding: utf-8 -*- 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

class NewsService():
    def __init__(self):
        self.stock_code = None

    def new_model(self):
        print(f'ENTER STEP 1 : new_model ')
        stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                       header=0)[0]
        stock_code.종목코드=stock_code.종목코드.map('{:06d}'.format)
        stock_code=stock_code[['회사명','종목코드']]

        stock_code=stock_code.rename(columns={'회사명':'company','종목코드':'code'})
        #stock_code.to_csv('/Users/YoungWoo/stock_crawl/company.csv', index=False, encoding='UTF-8')
        #code_df.head()
        self.stock_code = stock_code

    def search_news(self,companyname):
        print(f'ENTER STEP 2 : search_news ')
        print(f'companyname : {companyname}')

        article_result =[]
        title_result = []
        link_result = []
        date_result = []
        #source_result = []
        #stock_result = []

        stock_code = self.stock_code
        plusUrl = companyname.upper()
        plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()
        
        
       
        #94,174 5,33
        for i in range(2,4):

            url = 'https://finance.naver.com/item/news_news.nhn?code='+ str(plusUrl)+'&page={}'.format(i)
            print(f'url : {url}')
            source_code = requests.get(url).text
            html = BeautifulSoup(source_code, "lxml")
            rprts = html.findAll("table", {"class":"type5"})

            for items in rprts:
                #stock_result.append(plusUrl.zfill(6))
                
                titles = items.select(".title")
                #print(titles)
                #title_result=[]
                for title in titles: 
                    title = title.get_text() 
                    title = re.sub('\n','',title)
                    title_result.append(title)

                article = items.select('.title')
                #article_result =[]
                #print(article_result)
                for li in article:
                    lis =  'https://finance.naver.com' + li.find('a')['href']
                    articles_code = requests.get(lis).text
                    htmls = BeautifulSoup(articles_code,"lxml")
                    #docs = htmls.find("table",{"class":"view"})
                    docs = htmls.find("div",{"class":"scr01"})
                    docs = docs.text.replace('/','').replace('?','').replace("\t",'').replace("\n",'').replace('/n','').replace('[','').replace(']','').replace('!','').replace('-','').replace('$','').replace('▲','').replace("'",'').replace('■','').replace('◆','').replace('#','').replace('_','').replace('=','').replace('"','').replace(" \'",'').replace('아웃링크','').replace('◀','').replace('▶','').replace('<','').replace('>','').replace(':','').replace(',','').replace('ⓒ','').replace('※','').replace('\xa0','').replace('&','').replace('△','').replace('이데일리','').replace('매일경제','').replace('파이낸셜뉴스','').replace('서울경제','').replace('한국경제','').replace('조선비즈','').replace('아시아경제','').replace('머니투데이','').replace('헤럴드경제','').replace('···','').replace('·','').replace('‘','').replace('’','').replace('..','').replace("“",'').replace("”",'').replace('`','').replace('…','').replace('Copyrights','').replace('━','').lstrip()
            
                    #docs = docs.get_text()
                

                    #print(type(docs))
                    #print(docs)
                    article_result.append(docs)

                links = items.select('.title') 
                #link_result =[]
                #print(link_result)
                for link in links: 
                    add = 'https://finance.naver.com' + link.find('a')['href']
                    #print(add)
                    link_result.append(add)

                dates = items.select('.date') 
                #date_result = [date.get_text() for date in dates] 
                for date in dates:
                    date = date.get_text()
                    date_result.append(date)
                #print(date_result)

                # sources = items.select('.info')
                # #source_result = [source.get_text() for source in sources]
                # for source in sources:
                #     source = source.get_text()
                #     source_result.append(source)
                # #print(source_result)


            result= {"date" : date_result, "headline" : title_result, "contents" : article_result, "url" : link_result,"stock":plusUrl.zfill(6)} 
            # press" : source_result
            df_result = pd.DataFrame(result)
            return df_result
                        
            # print("다운 받고 있습니다------")
            # df_result.to_csv(str(plusUrl)+ '.csv', index=False, encoding='utf-8-sig') 

print('============= START ==================')        
service = NewsService()
service.new_model()
# print(service.stock_code.head())
df_result = service.search_news('lg이노텍')
# print(df_result.head())
# print(df_result.columns)
# print(df_result['contents'][2])
# print(df_result)
