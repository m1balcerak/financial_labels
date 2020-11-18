import pandas as pd
from data.sql.sql_syntax import SqlSyntax
import data.labels as lb
import numpy as np
from data.config_reader import config_reader
from typing import List, Dict, Tuple

def get_min_avg_prices(data_start,
               data_stop,
               sql,
               market):
    rows = sql.select_min_price_by_time(table=sql.initiate_market_minutes_table('{}_minutes'.format(market)),
                                        from_time=data_start,
                                        to_time=data_stop)
    df = pd.DataFrame(data=rows, columns=["Price"])
    return df['Price'].values

def get_sql_path(exchange):
    if exchange == 'binance':
        return 'sqlite:///../data/storage/binance.db'

def get_sql_syntax(exchange):
    return SqlSyntax(db_url=get_sql_path(exchange=exchange))

def get_min_data(data_start,
                    data_stop,
                    sql,
                    market):
    rows = sql.select_min_by_time(table=sql.initiate_market_minutes_table('{}_minutes'.format(market)),
                                    from_time=data_start,
                                    to_time=data_stop)

    min_data = pd.DataFrame(data=rows, columns=['trade_date', 'Price_low', 'Price_high', 'Sell_Volume', 'Buy_Volume', 'Price', 'Open', 'Closed'])

    return min_data



def get_coef_array(config,prices):

    if config.get_labels_parameters()['labels_version'] == 'up_down':
        return lb.get_coef_array_up_down(
                        prices=prices,
                        coefficient_power=config.get_labels_parameters()['coefpower'],
                        up_limit=config.get_labels_parameters()['up_limit'],
                        down_limit=config.get_labels_parameters()['down_limit'],
                        n_limits=config.get_labels_parameters()['n_limits'],
                        window=config.get_labels_parameters()['window'])
    else:
        if config.get_labels_parameters()['labels_version'] == 'threshold':
            return lb.get_coef_array_threshold(prices=prices,
                                        distance=config.get_labels_parameters()['distance'],
                                        buy_threshold=config.get_labels_parameters()['buy_threshold'],
                                        sell_threshold=config.get_labels_parameters()['sell_threshold'],
                                        coefficient_power=config.get_labels_parameters()['coefpower'],
                                        window=config.get_labels_parameters()['window'],
                                        normalise=True) 
        else:
            raise AttributeError('unsupported label version')

class data_reader(object):
    def __init__(self,config : config_reader):
        self.config = config
        self.sql_train = None
        self.sql_test  = None
        self.prices_train = None
        self.prices_test = None

        self.min_data_train = None
        self.min_data_test = None
        self.labels_train: np.array = None
        self.labels_test: np.array = None

        self.coef_array_train: np.array = None
        self.coef_array_test: np.array = None
        
        self.loss_scaler_train: np.array = None
        self.loss_scaler_test: np.array = None

    def release_raw_data(self):
        del self.coef_array_test
        del self.coef_array_train
        del self.prices_test
        del self.prices_train
        del self.min_data_train
        del self.min_data_test
        del self.sql_test
        del self.sql_train
        self.sql_train = None
        self.sql_test = None
        self.min_data_test = None
        self.min_data_train = None
        self.prices_train = None
        self.prices_test = None
        self.coef_array_train = None
        self.coef_array_test = None
    
    def get_labels_train(self):
        if self.labels_train is None:
            self.labels_train, self.loss_scaler_train = lb.make_training_data(
                            boundary=self.config.get_labels_parameters()['boundary'],
                            coef_array_sum=self.get_coef_array_train())
        return self.labels_train   

    def get_loss_scaler_train(self):
        if self.loss_scaler_train is None:
            self.labels_train, self.loss_scaler_train = lb.make_training_data(
                            boundary=self.config.get_labels_parameters()['boundary'],
                            coef_array_sum=self.get_coef_array_train())
        return self.loss_scaler_train    

    def get_labels_test(self):
        if self.labels_test is None:
            self.labels_test, self.loss_scaler_test = lb.make_training_data(
                            boundary=self.config.get_labels_parameters()['boundary'],
                            coef_array_sum=self.get_coef_array_test())
        return self.labels_test   

    def get_loss_scaler_test(self):
        if self.loss_scaler_test is None:
            self.labels_test, self.loss_scaler_test = lb.make_training_data(
                            boundary=self.config.get_labels_parameters()['boundary'],
                            coef_array_sum=self.get_coef_array_test())
        return self.loss_scaler_test   

    def get_sql_train(self):
        if self.sql_train is None:
            self.sql_train = get_sql_syntax(self.config.get_data_parameters()['exchange_train'])
        
        return self.sql_train

    def get_sql_test(self):
        if self.sql_test is None:
            self.sql_test = get_sql_syntax(self.config.get_data_parameters()['exchange_test'])
        
        return self.sql_test
    
    def get_min_data_train(self):
        if self.min_data_train is None:
            self.min_data_train = get_min_data(data_start=self.config.get_data_parameters()['date_start_train'],
                                data_stop=self.config.get_data_parameters()['date_end_train'],
                                sql=self.get_sql_train(),
                                market=self.config.get_data_parameters()['market_train'])
        return self.min_data_train

    def load_min_data_train(self, min_data_train):
        self.min_data_train = min_data_train

    def load_min_data_test(self, min_data_test):
        self.min_data_test = min_data_test

    def get_min_data_test(self):
        if self.min_data_test is None:
            self.min_data_test = get_min_data(data_start=self.config.get_data_parameters()['date_start_test'],
                                data_stop=self.config.get_data_parameters()['date_end_test'],
                                sql=self.get_sql_test(),
                                market=self.config.get_data_parameters()['market_test'])
        return self.min_data_test

    
    def get_prices_train(self):
        if self.prices_train is None:
            self.prices_train = get_min_avg_prices(data_start=self.config.get_data_parameters()['date_start_train'],
                                data_stop=self.config.get_data_parameters()['date_end_train'],
                                sql=self.get_sql_train(),
                                market=self.config.get_data_parameters()['market_train'])
        return self.prices_train

    def get_prices_test(self):
        if self.prices_test is None:
            self.prices_test = get_min_avg_prices(data_start=self.config.get_data_parameters()['date_start_test'],
                                data_stop=self.config.get_data_parameters()['date_end_test'],
                                sql=self.get_sql_test(),
                                market=self.config.get_data_parameters()['market_test'])
        return self.prices_test


    def get_coef_array_train(self):
        if self.coef_array_train is None:
            self.coef_array_train = get_coef_array(config=self.config,
                                                    prices=self.get_prices_train())

        return self.coef_array_train
    
    def get_coef_array_test(self):
        if self.coef_array_test is None:
            self.coef_array_test = get_coef_array(config=self.config,
                                                    prices=self.get_prices_test())

        return self.coef_array_test