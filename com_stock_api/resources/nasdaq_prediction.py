from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.resources.uscovid import USCovidDto, USCovids, USNewCases, CANewCases
from com_stock_api.resources.investingnews import InvestingDto, AppleSentiment, TeslaSentiment
from com_stock_api.resources.yhfinance import YHFinanceDto, AppleGraph, TeslaGraph
from sqlalchemy import and_,or_,func, extract
import os
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
import json
from datetime import datetime, timedelta
from sklearn import preprocessing
import numpy as np
from com_stock_api.utils.file_helper import FileReader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.python.keras.backend as K
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score



os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

from tqdm import tqdm

# =============================================================
# =============================================================
# ===================      Modeling    ========================
# =============================================================
# =============================================================

class NasdaqPredictionVo:
    id: int = 0
    ticker: str = ''
    date : str = ''
    pred_price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    compound : float = 0.0
   

# =============================================================
# =============================================================
# =================      Deep Learning    =====================
# =============================================================
# =============================================================

#PRE: Create DB, Insert All preprocessed data into DB
class NasdaqDF():

    path : str 
    tickers : str
    # cols : ticker,date,open,high,low,close,adjclose,volume
    def __init__(self):
        self.path = os.path.abspath(__file__+"/.."+"/saved_data")
        self.fileReader = FileReader()
        self.df = None
        self.ticker = ''
        self.tickers= ['AAPL', 'TSLA']

    def hook(self):
        
        for tic in self.tickers:
            self.ticker = tic
            df = self.create_dataframe()
            # self.scatter_graph()
            # self.heatmap_graph()
            # self.draw_initial_graph(df)

    #Step1. Collect all the features which would affect on prediction to the one CSV file 
    def create_dataframe(self):
        main_df = self.df
        
        #1. Bring history of a chosen ticker
        df=pd.read_sql_table('yahoo_finance', engine.connect())
        df = df.loc[(df['ticker'] == self.ticker) & (df['date'] > '2009-12-31')& (df['date'] < '2020-07-01')]
        df = df.drop(['ticker', 'id'], axis=1)

        #1-1. Adding three more features: moving average, increase in volume(%), increase in adj_close(%)
        avg = []
        df['moving_avg'] = df['adjclose'].rolling(window=50, min_periods=0).mean()
        
        i =1
        vol_rate = [0]
        adjclose_rate = [0]

        while i < len(df):
            vol_rate.append((df.iloc[i]['volume']- df.iloc[i-1]['volume'])/df.iloc[i-1]['volume'])
            adjclose_rate.append((df.iloc[i]['adjclose']-df.iloc[i-1]['adjclose'])/df.iloc[i-1]['adjclose'])
            i+=1
        
        df['increase_rate_vol'] = vol_rate
        df['increase_rate_adjclose'] =  adjclose_rate 
       
        #1-2. Bring history of the other ticker
        df4=pd.read_sql_table('yahoo_finance', engine.connect())
        tic = [t for t in self.tickers if t != self.ticker]
        df4 = df4.loc[(df4['ticker'] == tic[0]) & (df4['date'] > '2009-12-31')& (df4['date'] < '2020-07-01')]
        df4 = df4.rename(columns={'open': tic[0]+'_open', 'high':tic[0]+'_high', 'low':tic[0]+'_low','close': tic[0]+'_close', 'adjclose': tic[0]+'_adjclose'})
        df4 = df4.drop(['ticker', 'id', 'volume'], axis=1)


        temp_df = pd.DataFrame()
        df5 = pd.read_sql_table('korea_finance', engine.connect())
        k_tickers = {'lgchem': '051910', 'lginnotek':'011070'} #LG Chem & LG Innotek 
        for k_tic, v in k_tickers.items():
            df5 = pd.read_sql_table('korea_finance', engine.connect())
            
            df5['date']= pd.to_datetime(df5['date'])
            df5 = df5.loc[(df5['ticker'] == v) & (df5['date'] > '2009-12-31')& (df5['date'] < '2020-07-01')]
            df5 = df5.rename(columns={'open': k_tic+'_open', 'close':  k_tic+'_close', 'high': k_tic+'_high', 'low': k_tic+'_low'})
            df5 = df5.drop(['id','ticker','volume'], axis=1)
            df5 = df5[df5['date'].notnull() == True].set_index('date')

            temp_df = temp_df.join(df5, how='outer')
  
        temp_df['date'] = temp_df.index


        #2. Bring news sentiment 
        if (self.ticker == 'AAPL'):
            apple_json = json.dumps(AppleSentiment.get()[0], default = lambda x: x.__dict__)
            df2 = pd.read_json(apple_json)
            df2 = df2.drop(['id', 'pos', 'neg' ,'neu'], axis=1)

        else:
            tesla_json = json.dumps(TeslaSentiment.get()[0], default = lambda x: x.__dict__)
            df2 = pd.read_json(tesla_json)
            df2 = df2.drop(['id', 'pos', 'neg' ,'neu'], axis=1)

        #3. Bring US_covid status per day
        covid_json = json.dumps(USCovids.get()[0], default = lambda x: x.__dict__)
        df3 = pd.read_json(covid_json)
        df3 = df3.loc[(df3['date'] > '2019-12-31') & (df3['date'] < '2020-07-01')]
        df3 = df3.drop(['id', 'total_cases', 'total_deaths', 'ca_cases', 'ca_deaths'], axis=1)

        #4. Combine all features in one csv file
        df = df[df['date'].notnull() == True].set_index('date')
        df4 = df4[df4['date'].notnull() == True].set_index('date')
        temp_df = temp_df[temp_df['date'].notnull()==True].set_index('date')

        df2 = df2[df2['date'].notnull() == True].set_index('date')
        df3 = df3[df3['date'].notnull() == True].set_index('date')

        if self.ticker == 'AAPL':
            main_df = df.join(df4, how='outer')
        else:
            main_df = df.join(df4, how='inner')

        main_df = main_df.join(temp_df, how='outer')
        main_df = main_df.join(df2, how='outer')
        main_df = main_df.join(df3, how='outer')
        main_df[['new_us_cases', 'new_us_deaths', 'new_ca_cases', 'new_ca_deaths']] = main_df[['new_us_cases', 'new_us_deaths', 'new_ca_cases', 'new_ca_deaths']].fillna(value=0)
       
       # Fill Nan values in stock proices with interpolated values
        main_df = main_df.astype(float).interpolate()
        main_df = main_df.fillna(value=0)

        #6. Save to CSV file
        output_file = self.ticker + '_dataset.csv'
        result = os.path.join(self.path, output_file)
        main_df.to_csv(result)

        print(main_df)
        return main_df

        '''
  Index(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'moving_avg',
       'increase_rate_vol', 'increase_rate_adjclose', 'AAPL_open', 'AAPL_high',
       'AAPL_low', 'AAPL_close', 'AAPL_adjclose', 'lg_chem_open',
       'lg_chem_close', 'lg_chem_high', 'lg_chem_low', 'lg_innotek_open',
       'lg_innotek_close', 'lg_innotek_high', 'lg_innotek_low', 'neg', 'pos',
       'neu', 'compound', 'new_us_cases', 'new_us_death', 'new_ca_cases',
       'new_ca_death'],
        '''

    def draw_initial_graph(self, df):
        
        # df=pd.read_sql_table('yahoo_finance', engine.connect(), parse_dates=['date'])
        # df = df.loc[(df['ticker'] == self.ticker) & (df['date'] > '2009-12-31')& (df['date'] < '2020-07-01')]
        
        cols = ['open', 'low', 'close', 'adjclose' , 'moving_avg', 'increase_rate_vol', 'increase_rate_adjclose']
        path = os.path.abspath(__file__+"/.."+"/plots/")
        
        # Getting only business days
        isBusinessDay = BDay().onOffset
        df['date'] = df.index
        match_series = pd.to_datetime(df['date']).map(isBusinessDay)
        df = df[match_series]
        print(df)        
        for c in cols:
            title = self.ticker + " " + c + " graph"
            ax = df.plot(x='date', y=c, colormap='Set2', title = title)
            if 'rate' in c:
                ax.set(xlabel="Date", ylabel="percentage (%)")
                ax.hlines(y=0, xmin='2020-01-01', xmax='2020-07-01', colors='r', linestyles='--', lw=2)
            else:
                ax.set(xlabel="Date", ylabel="$ U.S. Dollar")
            # plt.show()
            file_name = title + ".png"
            output_file = os.path.join(path, file_name)
            plt.savefig(output_file)

    def scatter_graph(self):
        path = os.path.abspath(__file__+"/.."+"/plots/")        
        filen = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, filen)
       
        df = pd.read_csv(input_file)
        df.drop(['date'], axis=1)
        tic = [t for t in self.tickers if t !=self.ticker]
        op_tic = tic[0]

        sns.pairplot(df, height = 2.5)
        plt.tight_layout()
        plt.title("The Scatter Plot of "+ self.ticker)
        file_name = self.ticker + "_correlation.png"
        output_file = os.path.join(path, file_name)
        plt.savefig(output_file)
        print('=== Saved scatter plot ===')
        

    def heatmap_graph(self):

        filen = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, filen)
        df = pd.read_csv(input_file, header=0)
        df.drop(['date'], axis=1, inplace=True)
        print(df.columns)
        tic = [t for t in self.tickers if t !=self.ticker]
        op_tic = tic[0]

        sns.heatmap(df)
        plt.title('Heatmap of ' + self.ticker, fontsize=20)
        
        path = os.path.abspath(__file__+"/.."+"/plots/")
        file_name2 = self.ticker + "_heatmap.png"
        output_file2 = os.path.join(path, file_name2)
        plt.savefig(output_file2)
        print('=== Saved heatmap ===')

    
class LongShortTermModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.path = os.path.abspath(__file__+"/.."+"/saved_data")
        if self.ticker == 'AAPL':
            self.model_path = os.path.abspath(__file__+"/.."+"/models/apple")
        else:
            self.model_path = os.path.abspath(__file__+"/.."+"/models/tesla")
        self.features = None
        self.timesteps = None

    def hook(self):
        train_X, train_y, test_X, test_y = self.getting_data()
        self.save_model(train_X, train_y, test_X, test_y)
        self.eval_model(train_X, test_X, test_y)
        
    def getting_data(self):
        filen = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, filen)
        #load dataset
        df = pd.read_csv(input_file, header=0)
        df = df.drop(['close', 'volume'], axis=1)
       
        # get a list of columns
        cols = list(df)
       
        # move the column to tail of list using index, pop and insert
        cols.insert(25, cols.pop(cols.index('adjclose')))
        df = df.loc[:, cols] 
        df = df.set_index('date')
        values = df.values
       
        #normalize data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(values)

        #Split into train and test sets
        n_train_days = 20 * 12 * 10 # one month in business days = 20 days
        train = scaled[:n_train_days, :]
        test = scaled[n_train_days:, :]

        #split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        #reshape input to be 3D
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # (2400, 1, 25) (2400,) (321, 1, 25) (321,)

        self.features = train_X.shape[2]
        self.timesteps = train_X.shape[1]
        
        return train_X, train_y, test_X, test_y

    def create_model(self):
       
        #design network
        num_classes = 10
        batch_size = 32 

        model = Sequential()

        drop_out = 0.3
        lr=0.0001
        model.add(LSTM(units=600, input_shape=(self.timesteps, self.features)))
        model.add(Dense(units=200, activation='tanh'))
        model.add(Dropout(drop_out))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=lr, amsgrad=True), loss='mse', metrics=['mae', 'mse']) 

        model.summary()
        return model



    def save_model(self, train_X, train_y, test_X, test_y):

        checkpoint_path = os.path.join(self.model_path, 'checkpoint.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)

        model = self.create_model()
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1)

        #fit network for APPLE
        # history=model.fit(train_X, train_y, epochs=100, batch_size=200, validation_data=(test_X, test_y), verbose=1, shuffle=False, callbacks=[cp_callback])
        
        #fit network for TESLA
        history=model.fit(train_X, train_y, epochs=75, batch_size=200, validation_data=(test_X, test_y), verbose=1, shuffle=False, callbacks=[cp_callback])
        #plot history
        model.save(checkpoint_path)
        y_pred = model.predict(test_X,verbose=1, batch_size=100)
        print(y_pred[:5])
        print('expected: ', test_y[:5])

        plt.plot(test_y, label="price-original")
        plt.plot(y_pred, label="price-predicted")
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        

    def eval_model(self,train_X, test_X, test_y):

        checkpoint_path = os.path.join(self.model_path, 'checkpoint.h5')

        new_model = keras.models.load_model(checkpoint_path)
        new_model.summary()

        loss, mae, mse = new_model.evaluate(test_X,  test_y, verbose=2)
        print("NEW MODEL Mean Absolute Error after loading weights: ${:5.2f} ".format(mae))


        #test for apple
        x = [[118.64, 117.62, 117.08, 117.5 , 0.01, 0,503.50,522.22, 501.79, 519.58, 519.58, 739000, 724000, 748000, 735000, 164000, 163000, 164000, 161000,0 , 13904, 39, 1784, 2]]
        #test for tesla
        y =[[503.50,522.22, 501.79, 510.5 , 0.05, 0,118.64, 117.62, 117.08, 113.88, 113.88, 739000, 724000, 748000, 735000, 164000, 163000, 164000, 161000,0 , 13904, 39, 1784, 2]]
        #normalize data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(x)
        X = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

        price = new_model.predict(X)
        scaled[0,0]=price[0,0]
        unscaled = scaler.inverse_transform(scaled)
        print('price:' ,unscaled[0,0])

    
# if __name__ == "__main__":
#     # lstm = LongShortTermModel('AAPL')
#     lstm = LongShortTermModel('TSLA')
#     lstm.hook()
    
      
# =============================================================
# =============================================================
# ===================      Resourcing    ======================
# =============================================================
# =============================================================

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('pred_price', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('open', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('high', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('low', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('compound', type=str, required=False, help='This field cannot be left blank')
        
class Prediction(Resource):

    @staticmethod
    def post():
        print("=====nasdaq_prediction.py / ApplePredGraph's post")
        
        parser.add_argument('ticker', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('open', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('high', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('low', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('compound', type=str, required=True,
        help='This field should be a number')
    
        print('======== WE ARE IN THE PREDICTION =======')
        service = Service()
        args = parser.parse_args()
        apple = NasdaqPredictionVo()
        apple.ticker = args.ticker
        apple.open = args.open
        apple.high = args.high
        apple.low = args.low
        print('=========Received values ======')
        print(args.ticker, ' , ', args.open, ' , ', args.high, ' , ', args.low, ', ',args.compound)
        apple.compound = args.compound
        service.assign(apple)
        price = service.predict()
        # price = 150
        print(f'Predicted adjust close price is $ {price}')
        return json.dumps({'price': str(price)}), 200
        
Session = openSession()
session = Session()
global graph
graph = tf.compat.v1.get_default_graph()

class Service(object):
    
    open : float = 0.0
    high : float = 0.0
    low : float = 0.0
    compound : float = 0.0
    ticker :str = ''

    def __init__(self):
        if self.ticker == 'AAPL' or self.ticker == 'aapl':
            self.path = os.path.abspath(__file__+"/.."+"/models/apple/")
        else:
            self.path = os.path.abspath(__file__+"/.."+"/models/tesla/")

    def assign(self, param):
        self.open = float(param.open)
        self.high = float(param.high)
        self.low = float(param.low)
        self.compound = float(param.compound)
        self.ticker = param.ticker

    def predict(self):
        K.clear_session()
        
        checkpoint_path = os.path.join(self.path, 'checkpoint.h5')
        

        with graph.as_default():
            model = keras.models.load_model(checkpoint_path)
            model.summary()
            
            print('====================5===========================')

        ######### PREPARE FOR PREDICTION FEATURES ###############
            x = []
            x.append(self.open)
            x.append(self.high)
            x.append(self.low)
            x.append((self.high + self.low)/2) #'moving_avg'
            x.append(0.05)  #'increase_rate_vol'
            x.append(0.05)  #'increase_rate_adjclose'
            
            #the other NASDAQ stock price from the MarisDB
            apple_df = pd.read_sql_table('yahoo_finance', engine.connect())
            op_tic = ['TSLA', 'AAPL']
            tic = [t for t in op_tic if t != self.ticker]
            apple_df = apple_df.loc[(apple_df['ticker']==tic[0])].iloc[-1]

            x.append(apple_df['open'])
            x.append(apple_df['high'])
            x.append(apple_df['low'])
            x.append(apple_df['close'])
            x.append(apple_df['adjclose'])

            #the other KOSPI stocks pricees from the MarisDB
            KOSPI = {'lgchem': '051910', 'lginnotek':'011070'}

            for k_tic, v in KOSPI.items():
                kospi_df = pd.read_sql_table('korea_finance', engine.connect())
                kospi_df = kospi_df.loc[(kospi_df['ticker'] == v)].iloc[-1]
                x.append(float(kospi_df['open']))
                x.append(float(kospi_df['close']))
                x.append(float(kospi_df['high']))
                x.append(float(kospi_df['low']))
            
            x.append(self.compound)

            #Append covid cases 
            covid_json = json.dumps(USCovids.get()[0], default = lambda x: x.__dict__)
            covid_df = pd.read_json(covid_json)
            covid_df = covid_df.iloc[-1]
            
            x.append(covid_df['new_us_cases'])
            x.append(covid_df['new_us_deaths'])
            x.append(covid_df['new_ca_cases'])
            x.append(covid_df['new_ca_deaths'])

            x = [x]
            
            #normalize data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(x)
            X = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

            #predict the price and back to actual price
            price = model.predict(X)
            scaled[0,0]=price[0,0]
            unscaled = scaler.inverse_transform(scaled)
        
        return round(unscaled[0,0], 2)


# if __name__ == "__main__":
    # test = Prediction()
    # service = Service()
    # apple = NasdaqPredictionVo()
    # apple.open = 117.62
    # apple.high = 118.64
    # apple.low = 117.08
    # apple.compound = -0.5
    # apple.ticker = 'AAPL'
    # service.assign(apple)
    # service.predict()