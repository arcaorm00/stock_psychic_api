# -*- coding: utf-8 -*- 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re


class StockService():

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


    def serach_stock(self,companyname):
        result=[]
        
        stock_code = self.stock_code
        plusUrl = companyname.upper()
        plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()
        #print(plusUrl)
        
        def refine_price(text):
            price=int(text.replace(",",""))
            return price
            #print(type(price))

        for i in range(1,90):
            url='https://finance.naver.com/item/sise_day.nhn?code='+str(plusUrl)+'&page={}'.format(i)
            response=requests.get(url)
            text=response.text
            html=BeautifulSoup(text,'html.parser')
            table0=html.find_all("tr",{"onmouseover":"mouseOver(this)"})
            #print(url)
            for tr in table0:
                date= tr.find_all('td')[0].text
                
                temp=[]  
                
                for idx,td in enumerate(tr.find_all('td')[1:]):
                    if idx==1:
                        try:
                            #print(td.find()['alt'])
                            temp.append(td.find()['alt'])
                        except: 
                            temp.append('')
                
                    price=refine_price(td.text)
                    #print(price)
                    temp.append(price)
        
                #print([date]+temp)
                result.append([date]+temp)
        #print(result)

        df_temp=pd.DataFrame(result,columns=['date','close','up/down','pastday','open','high','low','volume']) #'날짜','종가','상승/하락','전일대비','시가','고가','저가','거래량'
        #df_temp
        df_temp.drop(['up/down', 'pastday'], axis='columns', inplace=True)
        df_temp['stock']=plusUrl
        return df_temp


service = StockService()
service.new_model
df_result = service.serach_stock('lg이노텍')
print(df_result.head())
