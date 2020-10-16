import os
from dataclasses import dataclass
import pandas as pd
import json

@dataclass
class FileReader:
    context: str = ''
    fname:str = ''
    train:object = None
    test:object = None
    id:str = ''
    label:str = ''

    def new_file(self):
        return os.path.join(self.context, self.fname)
    
    def csv_to_dframe(self):
        file = self.new_file()
        return pd.read_csv(file, encoding='UTF-8', thousands=',')

    def xls_to_dframe(self, header, usecols):
        file = self.new_file()
        return pd.read_excel(file, header=header, usecols=usecols)
        # pandas version 1.x 이상 encoding='UTF-8' 불필요

    def json_load(self):
        file = self.new_file()
        return json.load(open(file, encoding='UTF-8'))