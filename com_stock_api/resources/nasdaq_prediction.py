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
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
# tf =  tf.compat.v1
# tf.disable_eager_execution()
from tqdm import tqdm

# =============================================================
# =============================================================
# ===================      Modeling    ========================
# =============================================================
# =============================================================

class NasdaqPredictionDto(db.Model):
    __tablename__ = 'NASDAQ_prediction'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key = True, index = True)
    ticker: str = db.Column(db.String(30))
    date: str = db.Column(db.Date)
    pred_price: float = db.Column(db.Float)
    open: float = db.Column(db.Float)
    high: float = db.Column(db.Float)
    low: float = db.Column(db.Float)
    close: float = db.Column(db.Float)
    stock_id: int = db.Column(db.Integer, db.ForeignKey(YHFinanceDto.id))
    covid_id : int = db.Column(db.Integer, db.ForeignKey(USCovidDto.id))
    news_id: int = db.Column(db.Integer, db.ForeignKey(InvestingDto.id))


    def __init__(self, ticker, date, pred_price, open, high, low, close, stock_id, covid_id, news_id):
        self.ticker = ticker
        self.date = date
        self.pred_price = pred_price
        self.open = open
        self.high = high
        self.low = low
        self.close = close

        self.stock_id = stock_id
        self.covid_id = covid_id
        self.news_id = news_id

    def __repr__(self):
        return f'NASDAQ_Prediction(id=\'{self.id}\',ticker=\'{self.ticker}\',date=\'{self.date}\', pred_price=\'{self.pred_price}\',\
                open=\'{self.open}\',high=\'{self.high}\',low=\'{self.low}\',close=\'{self.close}\',\
                stock_id=\'{self.stock_id}\',covid_id=\'{self.covid_id}\', news_id=\'{self.news_id}\' )'

    @property
    def json(self):
        return {
            'id' : self.id,
            'ticker' : self.ticker,
            'date' : self.date,
            'pred_price' : self.pred_price,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'stock_id' : self.stock_id,
            'covid_id' : self.covid_id,
            'news_id' : self.news_id
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commint()

class NasdaqPredictionVo:
    id: int = 0
    ticker: str = ''
    date : str = ''
    pred_price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    stock_id : str = ''
    covid_id : str = ''
    news_id : str = ''


Session = openSession()
session = Session()

class NasdaqPredictionDao(NasdaqPredictionDto):

    @staticmethod
    def count():
        return session.query(func.count(NasdaqPredictionDto.id)).one()

    @staticmethod
    def save(data):
        db.session.add(data)
        db.session.commit()
    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()

    @classmethod
    def delete(cls, id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
        
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @staticmethod   
    def bulk():
        tickers = ['AAPL', 'TSLA']
        for tic in tickers:
            path = os.path.abspath(__file__+"/.."+"/data/")
            file_name = tic + '_pred.csv'
            input_file = os.path.join(path,file_name)

            df = pd.read_csv(input_file)
            print(df.head())
            session.bulk_insert_mappings(NasdaqPredictionDto, df.to_dict(orient="records"))
            session.commit()
        session.close()

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']==stock.ticker]
        return json.loads(df.to_json(orient='records'))

    
    @classmethod
    def find_by_date(cls, date, tic):
        return session.query.filter(and_(cls.date.like(date), cls.ticker.ilike(tic)))
    @classmethod
    def find_by_ticker(cls, tic):
        print("In find_by_ticker")
   
        return session.query(NasdaqPredictionDto).filter((NasdaqPredictionDto.ticker.ilike(tic))).order_by(NasdaqPredictionDto.date).all()
        
    @classmethod
    def find_by_period(cls,tic, start_date, end_date):
        return session.query(NasdaqPredictionDto).filter(and_(NasdaqPredictionDto.ticker.ilike(tic),date__range=(start_date, end_date)))
    @classmethod
    def find_today_one(cls, tic):
        today = datetime.today()
        return session.query(NasdaqPredictionDto).filter(and_(NasdaqPredictionDto.ticker.ilike(tic),NasdaqPredictionDto.date.like(today)))


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
        
class Model:
    
    def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1,):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple = False
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

    
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

class NasdaqTrain():

    num_layers : int
    size_layer : int
    timestamp : int
    epoch : int
    dropout_rate : float
    future_day : int
    learning_rate : float
    
    @classmethod
    def __init__(self):
        self.num_layers = 1
        self.size_layer = 128
        self.timestamp =5
        self.epoch = 300
        self.dropout_rate = 0.8
        self.future_day = 30
        self.learning_rate = 0.01
        self.df =None

    @staticmethod
    def calculate_accuracy(real, predict):
        real = np.array(real) +1
        predict = np.array(predict) +1
        percentage = 1 - np.sqrt(np.mean(np.square((real-predict)/real)))
        return percentage * 100

    @staticmethod
    def anchor (signal, weight) :
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer   

    @classmethod
    def train(self, ticker):

        sns.set()
        tf.compat.v1.random.set_random_seed(1234)

        path = os.path.abspath(__file__+"/.."+"/saved_data")
        filen = ticker + "_dataset.csv"
        input_file = os.path.join(path, filen)
        df = pd.read_csv(input_file, header=0)
        df = df.dropna()
        self.df = df
        # df.drop(['date'], axis=1, inplace=True)
        # print(df.head())
        '''
              open     high       low     close  adjclose       volume  ...  lginnotek_low  compound  new_us_cases  new_us_death  new_ca_cases  new_ca_death
0      NaN      NaN       NaN       NaN       NaN          NaN  ...            NaN   0.74300           0.0           0.0           0.0           0.0
1  74.0600  75.1500  73.79750  75.08750  74.57300  135480400.0  ...       138500.0   0.93112           0.0           0.0           0.0           0.0
2  74.2875  75.1450  74.12500  74.35750  73.84800  146322800.0  ...       138000.0   0.00145           0.0           0.0           0.0           0.0
3  73.8675  75.0675  73.65625  74.65375  74.14225  132355000.0  ...       137000.0   0.97265           0.0           0.0           0.0           0.0
4  73.4475  74.9900  73.18750  74.95000  74.43650  118387200.0  ...       136000.0   0.90270           0.0           0.0           0.0           0.0

[5 rows x 27 columns]
        '''

        minmax = MinMaxScaler().fit(self.df.iloc[:,5:6].astype('float32')) #adjclose index
        df_log = minmax.transform(self.df.iloc[:,5:6].astype('float32'))
        df_log = pd.DataFrame(df_log)
        # print(df_log.head())
        '''
                  0
        0  0.525795
        1  0.505445
        2  0.513704
        3  0.521964
        4  0.512137
        '''
        #split train and test
        test_size = 30
        simulation_size = 10

        df_train = df_log.iloc[:-test_size]
        df_test = df_log.iloc[-test_size:]

        results = []
        for i in range(simulation_size):
            print('simulation %d' %(i + 1))
            results.append(self.forecast(df_train, df_log, test_size, self.df, minmax, i+1, ticker))

        # date_ori - pd.to_datetime(df.iloc[:, 0]).tolist()
        # for i in range(test_size):
        #     date_ori.append(date_ori[-1] + timedelta(days=1))
        # date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
        # print(date_ori[-5:])

        # accepted_results = []
        # for r in results:
        #     if ((np.array(r[-test_size:])) < np.min(df['adjclose']).sum() == 0 and \
        #         (np.array(r[-test_size:]) > np.max(df['adjclose']) * 2 ).sum()==0) :
        #         accepted_results.append(r)
        # print("length of accepted_results: ", len(accepted_results))
        return results

    def get_accuracies(self, results):
        return [self.calculate_accuracy(self.df['adjclose'].iloc[-test_size:].values, r) for r in results]
        

    def draw_forecasts(self, results):
        accuracies = self.get_accuracies(results)
        plt.figure(figsize= (15,5))
        for no, r in enumerate(results):
            plt.plot(r, label = 'forcast %d'% (no+1))
        plt.plot(self.df['adjclose'].iloc[-test_size:].values, label = 'true trend', c= 'black')
        plt.legend()
        plt.title('Average accuracy: %.4f'%(np.mean(accuracies)))

        path = os.path.abspath(__file__+"/.."+"/plots/")

        file_name = ticker + "_prediction.png"
        output_file = os.path.join(path, file_name)
        plt.savefig(output_file)

        print('==== Saved nasdaq.ckpy + prediction.png ====')

    @classmethod
    def forecast (self, df_train, df_log, test_size, df, minmax, cnt, ticker):
        tf.reset_default_graph()
        modelnn = Model(self.learning_rate, self.num_layers, df_log.shape[1], self.size_layer, 
        df_log.shape[1], self.dropout_rate)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        date_ori = pd.to_datetime(df.iloc[:,0]).tolist()

        pbar = tqdm(range(self.epoch), desc='train loop')
        for i in pbar:
            init_value = np.zeros((1, self.num_layers * 2 * self.size_layer))
            total_loss, total_acc = [], []
            for k in range(0, df_train.shape[0] - 1, self.timestamp):
                index = min(k + self.timestamp, df_train.shape[0]-1)
                batch_x = np.expand_dims(
                    df_train.iloc[ k : index, :].values, axis=0
                )
                batch_y = df_train.iloc[k + 1 : index +1 , :].values
                logits, last_state, _, loss = sess.run(
                    [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                    feed_dict = {
                        modelnn.X : batch_x,
                        modelnn.Y : batch_y,
                        modelnn.hidden_layer : init_value,
                    },
                )

                init_value = last_state
                total_loss.append(loss)
                total_acc.append(self.calculate_accuracy(batch_y[:, 0], logits[:,0]))
            pbar.set_postfix(cost = np.mean(total_loss), acc= np.mean(total_acc))

        future_day = test_size

        output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // self.timestamp) * self.timestamp
        init_value = np.zeros((1, self.num_layers * 2 * self.size_layer))

        for k in range(0, (df_train.shape[0] // self.timestamp) * self.timestamp, self.timestamp):
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state], 
                feed_dict = {
                    modelnn.X : np.expand_dims(
                        df_train.iloc[k: k + self.timestamp], axis = 0
                    ),
                    modelnn.hidden_layer : init_value,
                },
            )
            init_value = last_state
            output_predict[k+1 : k + self.timestamp + 1] = out_logits

        if upper_b != df_train.shape[0]:
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict = {
                    modelnn.X : np.expand_dims(df_train.iloc[upper_b:], axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            output_predict[upper_b +1 : df_train.shape[0]+1] = out_logits
            future_day -=1
            date_ori.append(date_ori[-1]+timedelta(days =1))

        init_value = last_state
        
        for i in range(future_day):
            o = output_predict[-future_day - self.timestamp + i : -future_day + i]
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict = {
                    modelnn.X : np.expand_dims(o, axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value =  last_state
            output_predict[-future_day + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days =1))

        output_predict = minmax.inverse_transform(output_predict)
        deep_future = self.anchor(output_predict[:, 0], 0.3)

        path2 = os.path.abspath(__file__+"/.."+"/models/")
        saver  = tf.train.Saver()
        name = '/' + ticker+ str(cnt) + '.ckpt'
        saver.save(sess, path2+name)

        return deep_future[-test_size:]

'''             
if __name__ == "__main__":
    # dataset = NasdaqDF()
    # dataset.hook()
    train = NasdaqTrain()
    NasdaqTrain.train('AAPL')
    NasdaqTrain.train('TSLA')
'''

# =============================================================
# =============================================================
# ===================      Resourcing    ======================
# =============================================================
# =============================================================

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=False, help='This field cannot be left blank')
parser.add_argument('pred_price', type=float, required=False, help='This field cannot be left blank')
parser.add_argument('stock_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('covid_id', type=int, required=False, help='This field cannot be left blank')
parser.add_argument('news_id', type=int, required=False, help='This field cannot be left blank')

class NasdaqPrediction(Resource):    
    @staticmethod
    def post():
        data = parser.parse_args()
        nprediction = NasdaqPredictionDto(data['date'], data['ticker'],data['pred_price'], data['stock_id'], data['covid_id'], data['news_id'])
        try: 
            nprediction.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200
        except:
            return {'message': 'An error occured inserting the article'}, 500
        return nprediction.json(), 201
    
    
    def get(self, id):
        article = NasdaqPredictionDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404

    def put(self, id):
        data = NasdaqPrediction.parser.parse_args()
        prediction = NasdaqPredictionDao.find_by_id(id)

        prediction.title = data['title']
        prediction.content = data['content']
        prediction.save()
        return prediction.json()

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Ticker {args["ticker"]} on date {args["date"]} is deleted')
        NasdaqPredictionDao.delete(args['id'])
        return {'code' : 0 , 'message' : 'SUCCESS'}, 200

class NasdaqPredictions(Resource):
    def get(self):
        return NasdaqPredictionDao.find_all(), 200
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}

class TeslaPredGraph(Resource):

    @staticmethod
    def get():
        print("=====nasdaq_prediction.py / TeslPredaGraph's get")
        stock = NasdaqPredictionVo
        stock.ticker = 'TSLA'
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====nasdaq_prediction.py / TeslaPredGraph's post")
        args = parser.parse_args()
        stock = NasdaqPredictionVo
        stock.ticker = args.ticker
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data[0], 200
        
class ApplePredGraph(Resource):

    @staticmethod
    def get():
        print("=====nasdaq_prediction.py / ApplePredGraph's get")
        stock = NasdaqPredictionVo
        stock.ticker = 'AAPL'
        data = NasdaqPredictionDao.find_all_by_ticker(stock)
        return data, 200


    @staticmethod
    def post():
        print("=====nasdaq_prediction.py / ApplePredGraph's post")
        
        parser.add_argument('open', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('high', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('low', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('close', type=str, required=True,
        help='This field should be a number')

        service = AppleService()
        args = parser.parse_args()
        apple = NasdaqPredictionVo()
        apple.ticker = 'AAPL'
        apple.open = args.open
        apple.high = args.high
        apple.low = args.low
        apple.close = args.close
        service.assign(apple)
        price = service.predict()
        print(f'Predicted adjust close price is $ {price}')
        return {'price': str(price)}, 200
        



class AppleService(object):
    
    open : float = 0.0
    high : float = 0.0
    low : float = 0.0
    close : float = 0.0
    

    def __init__(self):
        self.path = os.path.abspath(__file__+"/.."+"/models/apple/")

    def assign(self, param):
        self.open = param.open
        self.high = param.high
        self.low = param.low
        self.close = param.close
        self.ticker = param.ticker

    def predict(self):
        print('====in predict func ====')
        X = tf.placeholder(tf.float32, shape=[None, 3])
        W = tf.Variable(tf.random_normal([3,1]), name='weight')
        b = tf.Variable(tf.random_normal([1], name='bias'))
        saver = tf.train.Saver()
        result = []
        with tf.Session() as sess:
            saver.restore(sess, self.path+'/AAPL1.ckpt')
            sess.run(tf.global_variables_initializer())

            # fname = '/' + self.ticker + str(i+1) + '.ckpt'
            # print('File NAME: ', fname)
            # saver.restore(sess, self.path+fname)
            data = [[self.open, self.high, self.low], ]                
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X,W)+b, feed_dict={X:arr[0,3]})
            print ('dict!!' , dict[0])
        # avg_pred = sum(result)/len(result)
        return dict[0]

if __name__ == "__main__":

    service = AppleService()
    apple = NasdaqPredictionVo()
    apple.open = 120
    apple.high = 125
    apple.low = 118
    apple.ticker = 'AAPL'
    service.assign(apple)
    price = service.predict()
    print ("price is ", price)