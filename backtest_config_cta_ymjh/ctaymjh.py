# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 16:45
# @Author  : zhangfang


import pandas as pd
import numpy  as np
import datetime
from common.file_saver import file_saver
from common.decorator import runing_time
from strategy.strategy import Strategy
from data_engine.data_factory import DataFactory
from data_engine.global_variable import ASSETTYPE_FUTURE, DATASOURCE_REMOTE, DATASOURCE_LOCAL
from config.config import Config_back_test, Config_trading

from data_engine.instrument.future import Future
from data_engine.instrument.product import Product
from data_engine.market_tradingdate import Market_tradingdate
import data_engine.global_variable as global_variable
import talib



def get_trade_days(curr_date):
    trd_day_df = Market_tradingdate('SHFE').HisData_DF[['Tradedays_str', 'isTradingday']]
    trd_day_df = trd_day_df[trd_day_df['isTradingday'] == True]
    trd_day_after_curr = trd_day_df[trd_day_df['Tradedays_str'] > curr_date].Tradedays_str.iloc[0]
    return trd_day_df, trd_day_after_curr


def get_day_and_night_symbols(symbols_lst):
    night_symbols = []
    day_symbols = []
    for symbol in symbols_lst:
        ts = Future(symbol)#.get_trading_sessions()
        # tradeNightStartTime = ts.Session4_Start.zfill(8)
        # tradeNightEndTime = ts.Session4_End.zfill(8)
        # if tradeNightEndTime < '04:00:00' or tradeNightEndTime >= '23:30:00':
        #     tradeNightEndTime = '23:30:00'
        if ts.has_night_trading():
            night_symbols.append(symbol)
        else:
            day_symbols.append(symbol)
    return night_symbols, day_symbols


class CtaYmjhStrategy(Strategy):
    # _strategy_type = 'intraday_pair'
    def __init__(self, curr_date, asset_type, symbol, freq, s_period, l_period, maxLeverage, targetVol,
                 volLookback, period, result_fold, method, **kwargs):

        super(CtaYmjhStrategy, self).__init__(period=period,
                                              s_period=s_period,
                                              l_period=l_period,
                                              maxLeverage=maxLeverage,
                                              method=method,
                                              targetVol=targetVol,
                                              volLookback=volLookback,
                                              strategy_name='YMJH-Daily')
        self.aggToken = symbol
        self._freq = freq
        self._asset_type = asset_type
        self.symbol = symbol
        self.curr_date = curr_date
        # self.curr_date = '2020-01-16'
        self.method = method

        self._symbols = set()
        for s1 in [symbol]:
            if s1 not in self._symbols:
                self._symbols.add(s1)

        self._instrument = Future(symbol=symbol)
        self._product = Product(self._instrument.product_id)

        self._market_data = None
        self.result_fold = result_fold

    def _get_history(self, startDate, endDate, **kwargs):
        self._market_data = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq=self._freq,
                                                          symbols=self.symbol, end_date=endDate)

    def _get_history_daily(self, startDate, endDate):
        self._market_data_daily = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=self.symbol, start_date=startDate,
                                                                end_date=endDate)

    def get_trade_days(self):
        trd_day = Market_tradingdate('SHFE').HisData_DF[['Tradedays_str', 'isTradingday']]
        trd_day = trd_day[trd_day['isTradingday'] == True]
        self.trd_day = trd_day
        self.trd_day_after_curr = trd_day[trd_day['Tradedays_str'] > self.curr_date].Tradedays_str.iloc[0]

    def vol_estimator_garch(self, data_df, st=25, lt=252 * 3):
        st_vol = data_df.ewm(span=st, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        lt_vol = data_df.ewm(span=lt, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        decay_rate = 0.8
        vol = st_vol * decay_rate + lt_vol * (1 - decay_rate)
        return vol

    @staticmethod
    def get_resp_curve(x, method):
        resp_curve = pd.DataFrame()
        if method == 'gaussian':
            resp_curve = np.exp(-(x ** 2) / 4.0)
        return resp_curve

    def cap_vol_by_rolling(self, vol, target_vol):
        idxs = vol.index
        for idx in range(len(idxs)):
            curDate = idxs[idx]
            vol[curDate] = max(vol[curDate], target_vol)
        return vol

    def get_position(self, symbol, **kwargs):
        capital = kwargs['capital']
        targetVol = kwargs['targetVol']
        volLookback = kwargs['volLookback']
        N1_lst = self.s_period_dic[symbol]
        N2_lst = self.l_period_dic[symbol]
        capital_intial = capital / self._instrument.contract_size # self._contract_size_dict[symbol + '_VOL']
        signal_df_lst = []
        for i in range(len(N1_lst)):
            N1 = N1_lst[i]
            N2 = N2_lst[i]
            data = self._format_data(symbol, N1, N2, targetVol, volLookback)
            data.to_csv('e:/data_p.csv')
            _signal_lst = []
            _signal = 0
            for idx, _row in data.iterrows():
                condition_l = ((_row.ma_s1 == _row.ma_l1) and (_row.ma_s > _row.ma_l) and (
                        _row.ave_p >= _row.ma_s)) or ((_row.ma_s1 < _row.ma_l1) and (
                        _row.ma_s > _row.ma_l) and (_row.ave_p >= min(_row.ma_s, _row.ma_l)))
                condition_s = (_row.ma_s1 > _row.ma_l1) and (_row.ma_s < _row.ma_l) and (
                        _row.ave_p <= max(_row.ma_s, _row.ma_l))
                if _signal == 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                    elif condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal > 0:
                    if condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal < 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                _signal_lst.append(_signal)
            data[symbol + str(i)] = _signal_lst
            data[symbol + str(i)] = data[symbol + str(i)].fillna(0)
            signal_df_lst.append(data[symbol + str(i)])
        signal_df = pd.concat(signal_df_lst, axis=1)
        signal = np.mean(signal_df, axis=1)
        # data = dailyPrice.tail(1)

        position_trd = signal[-1]
        f = Future(data.contract_id.iloc[-1])
        instrument = f.ctp_symbol
        market = f.market
        data['date_time'] = data.index
        histLastSignalTime = data.date_time.iloc[-1]
        return position_trd, instrument, market, histLastSignalTime

    def gen_cofig(self, **kwargs):
        config_if_main_contract_switch = []
        config = Config_trading()

        kwargs['capital'] = self.aggtoken_capital(aggtoken=self.aggToken, date=global_variable.get_now())
        kwargs['targetVol'] = self.aggtoken_target_vol(aggtoken=self.aggToken)

        symbol = self.symbol
        fut = Future(symbol)
        p = Product(product_id=fut.product_id)
        p.load_hq()
        p.get_hq_panel()
        max_volume_fut = p.max_volume_fut()
        if self.method == 'night':
            print('night symbol {}'.format(symbol))
            ts = fut.get_trading_sessions()
            tradeNightStartTime = ts.Session4_Start.zfill(8)
            dailyPrice = self._market_data_daily[symbol]
            dailyPrice = dailyPrice.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))
            dailyPrice = dailyPrice[dailyPrice['trade_date'] <= self.curr_date]
            if str(dailyPrice.trade_date.iloc[-1])[:10] < self.curr_date:
                print('========Error: {} 未取得最新行情'.format(symbol))
                return config,config_if_main_contract_switch

            position_trd, instrument, market, histLastSignalTime = self.get_position(symbol, **kwargs)
            valideStartTime = self.curr_date + ' ' + tradeNightStartTime


            #换月
            if position_trd != 0:
                config_if_main_contract_switch = self.gen_config_if_main_contract_switch(product_obj=self._product,
                                                                                         curr_date_str=self.curr_date,
                                                                                         position_trd=0,
                                                                                         only_close_last_constract=True)

            if max_volume_fut is not None:
                instrument = max_volume_fut.ctp_symbol
            config = self.gen_target_position_config(requestType='Create',
                                                     instrument=instrument,
                                                     market=market,
                                                     aggToken=self.aggToken,
                                                     requestTime=valideStartTime,
                                                     aggregateRequest='true',
                                                     targetPosition=position_trd,
                                                     strategy=self._strategy_name,
                                                     histLastSignalTime=histLastSignalTime,
                                                     initiator='Agg-Proxy',
                                                         capital=kwargs['capital'],
                                                         targetVol=kwargs['targetVol'])

        elif self.method == 'day':
            print('day symbol {}'.format(symbol))
            ts = fut.get_trading_sessions()
            tradeDayStartTime = ts.Session1_Start.zfill(8)
            dailyPrice = self._market_data_daily[symbol]
            dailyPrice = dailyPrice.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))
            dailyPrice = dailyPrice[dailyPrice['trade_date'] <= self.curr_date]
            if str(dailyPrice.trade_date.iloc[-1])[:10] < self.curr_date:
                print('========Error: {} 未取得最新行情'.format(symbol))
                return config,config_if_main_contract_switch

            position_trd, instrument, market, histLastSignalTime = self.get_position(symbol, **kwargs)


            #换月
            if position_trd != 0:
                config_if_main_contract_switch = self.gen_config_if_main_contract_switch(product_obj=self._product,
                                                                                         curr_date_str=self.trd_day_after_curr,
                                                                                         position_trd=0,
                                                                                         only_close_last_constract=True)
            valideStartTime = self.trd_day_after_curr + ' ' + tradeDayStartTime
            if max_volume_fut is not None:
                instrument = max_volume_fut.ctp_symbol
            config = self.gen_target_position_config(requestType='Create',
                                                     instrument=instrument,
                                                     market=market,
                                                     aggToken=self.aggToken,
                                                     requestTime=valideStartTime,
                                                     aggregateRequest='true',
                                                     targetPosition=position_trd,
                                                     strategy=self._strategy_name,
                                                     histLastSignalTime=histLastSignalTime,
                                                     initiator='Agg-Proxy',
                                                         capital=kwargs['capital'],
                                                         targetVol=kwargs['targetVol'])
        return config,config_if_main_contract_switch

    @runing_time
    def gen_signal(self, symbol, **kwargs):
        capital = kwargs['capital']
        print(symbol,capital)
        # contract_size_list = self._contract_size_dict[symbol]
        capital_intial = capital / self._instrument.contract_size
        N1_lst = self.s_period_dic[symbol]
        N2_lst = self.l_period_dic[symbol]
        targetVol = self.aggtoken_target_vol(aggtoken=self.aggToken)  #kwargs['targetVol']
        volLookback = kwargs['volLookback']

        pos_df_list = []
        signal_df_lst = []
        for i in range(len(N1_lst)):
            N1 = N1_lst[i]
            N2 = N2_lst[i]
            data = self._format_data(symbol, N1, N2, targetVol, volLookback)
            _signal_lst = []
            _signal = 0
            for idx, _row in data.iterrows():
                condition_l = ((_row.ma_s1 == _row.ma_l1) and (_row.ma_s > _row.ma_l) and (
                        _row.ave_p >= _row.ma_s)) or ((_row.ma_s1 < _row.ma_l1) and (
                        _row.ma_s > _row.ma_l) and (_row.ave_p >= min(_row.ma_s, _row.ma_l)))
                condition_s = (_row.ma_s1 > _row.ma_l1) and (_row.ma_s < _row.ma_l) and (
                        _row.ave_p <= max(_row.ma_s, _row.ma_l))
                if _signal == 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                    elif condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal > 0:
                    if condition_s:
                        cost = _row.close
                        _signal = -_row.riskScaler * capital_intial / cost

                elif _signal < 0:
                    if condition_l:
                        cost = _row.close
                        _signal = _row.riskScaler * capital_intial / cost
                _signal_lst.append(_signal)
            data[symbol + str(i)] = _signal_lst
            data[symbol + str(i)] = data[symbol + str(i)].fillna(0)
            signal_df_lst.append(data[symbol + str(i)])
        # signal.to_csv('E://Strategy//OCM//signal_0.csv')
        signal_df = pd.concat(signal_df_lst, axis=1)
        signal = np.mean(signal_df, axis=1)

        target_pos_dict = {}
        target_pos_tmp = signal
        target_pos_tmp.fillna(method='pad', inplace=True)
        target_pos_tmp.name = symbol
        target_pos_dict[symbol] = target_pos_tmp
        if self.result_fold is not None:
            file_saver().save_file(target_pos_tmp, self.result_fold + '\\' + symbol + '_target_pos.csv')

        # 换月处理
        contract_id_series = data['contract_id']
        contract_id_series.name = symbol + '_contract_id'
        target_pos_df = pd.concat([target_pos_tmp, contract_id_series], axis=1)
        # target_pos_df.loc[
        #     target_pos_df[symbol + '_contract_id'] != target_pos_df[symbol + '_contract_id'].shift(-2), symbol] = 0
        target_pos_dict[symbol] = target_pos_df[symbol]

        pos_serires = target_pos_dict[symbol].copy()
        pos_serires.name = 'position'

        pos_df = pd.DataFrame(pos_serires, index=pos_serires.index)
        pos_df = pos_df.join(contract_id_series)
        pos_df['symbol'] = symbol
        pos_df['asset_type'] = self._asset_type
        pos_df['contract_size'] = self._contract_size_dict[symbol]
        pos_df['tick_size'] = self._tick_size_dict[symbol]
        pos_df['margin_ratio'] = 0.1
        pos_df['freq'] = self._freq
        pos_df['remark'] = '.'.join([self._strategy_name, self._strategy_type, symbol])

        pos_df_list.append(pos_df)
        signal_dataframe = None
        if len(pos_df_list) > 0:
            signal_dataframe = pd.concat(pos_df_list)
        return signal_dataframe

    @runing_time
    def _format_data(self, symbol, N1, N2, targetVol, volLookback):

        data_daily = self._market_data_daily[symbol]
        data_daily = data_daily[
            ['high', 'close', 'open', 'low', 'volume', 'trade_date', 'contract_id', 'price_return']]
        data_daily = data_daily[data_daily['volume'] > 0]
        data_daily = data_daily \
                         .assign(HH_s=lambda df: talib.MAX(df.high.values, N1)) \
                         .assign(LL_s=lambda df: talib.MIN(df.low.values, N1)) \
                         .assign(HH_l=lambda df: talib.MAX(df.high.values, N2)) \
                         .assign(LL_l=lambda df: talib.MIN(df.low.values, N2)) \
                         .assign(ma_s=lambda df: (df.HH_s + df.LL_s) / 2) \
                         .assign(ma_l=lambda df: (df.HH_l + df.LL_l) / 2) \
                         .assign(ma_s1=lambda df: df.ma_s.shift(1)) \
                         .assign(ma_l1=lambda df: df.ma_l.shift(1)) \
                         .assign(ave_p=lambda df: (2 * df.close + df.high + df.low) / 4) \
                         .loc[:,
                     ['contract_id', 'ma_s', 'ma_l', 'ma_s1', 'ma_l1', 'ave_p', 'high', 'low', 'close', 'open',
                      'trade_date', 'price_return']] \
            .rename(columns={'high': 'high_d', 'low': 'low_d', 'open': 'open_d'}) \
            .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x))) \
            .assign(trade_date_1=lambda df: df.trade_date.shift(1)) \
            .assign(close_d=lambda df: df.close.shift(1)) \
            .assign(trade_date_next=lambda df: df.trade_date.shift(-1))
        if volLookback != 0:
            realizedVol = data_daily['price_return'].ewm(
                span=volLookback, ignore_na=True, adjust=False).std(bias=True) * (252 ** 0.5)
            if symbol not in ['T_VOL', 'TF_VOL']:
                realizedVol = self.cap_vol_by_rolling(realizedVol, targetVol)
            riskScaler = targetVol / realizedVol
            data_daily['riskScaler'] = riskScaler
        else:
            data_daily['riskScaler'] = 1
        return data_daily

    @runing_time
    def run_test(self, startDate, endDate, **kwargs):
        self._parameter(**kwargs)
        self._get_market_info()
        self._get_history_daily(startDate=startDate, endDate=endDate)
        signal_dataframe = []
        for symbol in self._symbols:
            signal_dataframe.append(
                self.gen_signal(symbol=symbol, **kwargs))
        signal_dataframe = pd.concat(signal_dataframe)
        if self.result_fold is not None:
            signal_dataframe.to_csv(self.result_fold + '\\' + 'signal_dataframe.csv')
        return signal_dataframe

    def _parameter(self, **kwargs):
        symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM', ],
                        'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                        'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                        'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
                        'Equity': ['IF', 'IH', 'IC'],
                        'Bonds': ['T', 'TF'],
                        'PreciousMetal': ['AG', 'AU']}
        class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
        s_period_dict = kwargs['s_period']
        l_period_dict = kwargs['l_period']
        self.s_period_dic = {}
        self.l_period_dic = {}
        for clas in class_lst:
            symbols = symbols_dict[clas]
            for symbol in symbols:
                self.s_period_dic[symbol + '_VOL'] = s_period_dict[clas]
                self.l_period_dic[symbol + '_VOL'] = l_period_dict[clas]
                self.s_period_dic[symbol] = s_period_dict[clas]
                self.l_period_dic[symbol] = l_period_dict[clas]

    @runing_time
    def run_cofig(self, startDate, endDate, **kwargs):
        self._parameter(**kwargs)
        self._get_market_info()
        self._get_history_daily(startDate=startDate, endDate=endDate)
        self.get_trade_days()
        return self.gen_cofig(**kwargs)


class CtaYmjhStrategy_ex(CtaYmjhStrategy):
    def __init__(self, config, curr_date, asset_type, symbol, method='night'):
        assert isinstance(config, Config_back_test)
        CtaYmjhStrategy.__init__(self, curr_date=curr_date, asset_type=asset_type, symbol=symbol,
                                 freq=config.get_data_config('freq'),
                                 method=method, result_fold=config.get_result_config('result_folder')
                                 , **config.strategy_config)

        self._config = config

    @runing_time
    def run_cofig(self, **kwargs):
        return CtaYmjhStrategy.run_cofig(self, startDate=self._config.get_strategy_config('start_date'),
                                         endDate=self._config.get_strategy_config('end_date'), method='night',
                                         **self._config.strategy_config, **kwargs)

    @runing_time
    def run_test(self, **kwargs):
        print(self._config.get_strategy_config('end_date'))
        return CtaYmjhStrategy.run_test(self, startDate=self._config.get_strategy_config('start_date'),
                                        endDate=self._config.get_strategy_config('end_date'),
                                        **self._config.strategy_config, **kwargs)
