import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader
import pandas as pd
import numpy as np
import re

class BoardPro:

    def __init__(self):
        print(f'basedir: {basedir}')
        self.reader = FileReader()

    def process(self):
        file_data = self.get_data()
        data = self.refine_data(file_data)
        # self.save_data(data)

    def get_data(self):
        self.reader.context = os.path.join(basedir, 'data')
        self.reader.fname = 'korea_news_list.csv'
        notice_file = self.reader.csv_to_dframe()
        # print(notice_file)
        return notice_file
    
    @staticmethod
    def refine_data(data):
        # 컬럼명 변경
        data = data.rename({'제목': 'title', '내용': 'content', '작성일자': 'regdate'}, axis='columns')
        data['email'] = 'admin@admin.com'
        data = data.drop('url', axis=1)

        # print(data['content'][1])
        for idx in range(len(data['content'])):
            con = re.sub('<!--(.+?)-->', '', str(data['content'][idx]))
            con = con.replace('<!--', '')
            data['content'][idx] = con
        # data['regdate'] = ['20'+ regdate for regdate in data['regdate']]

        print(data)
        return data

    def save_data(self, data):
        self.reader.context = os.path.join(basedir, 'saved_data')
        self.reader.fname = 'korea_news_database.csv'
        data.to_csv(self.reader.new_file())
        print('file saved')

if __name__ == '__main__':
    b_pro = BoardPro()
    b_pro.process()