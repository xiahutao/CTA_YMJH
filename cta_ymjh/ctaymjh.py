# -*- coding: utf-8 -*-
"""
Created on Fri Dec  13 16:34:53 2019

@author: zhangfang
"""

import pandas as pd
import numpy  as np
import datetime
from common.file_saver import file_saver
from common.decorator import runing_time
from strategy.strategy import Strategy
from data_engine.global_variable import ASSETTYPE_FUTURE,DATASOURCE_REMOTE,DATASOURCE_LOCAL
from data_engine.data_factory import DataFactory
import talib
import math


class CtaYmjhStrategy(Strategy):
    def __init__(self, asset_type, symbols_list, freq, s_period, l_period, targetVol, volLookback,
                 period, result_fold, **kwargs):

        super(CtaYmjhStrategy, self).__init__(period=period,
                                             s_period=s_period,
                                             l_period=l_period,
                                              targetVol=targetVol,
                                              volLookback=volLookback
                                             )
        self._strategy_name = 'CtaYmjhStrategy'
        self._strategy_type = 'intraday_symbols'
        self._freq = freq
        self._asset_type = asset_type
        self._symbol_pair_list = symbols_list

        self._symbols = set()
        self._symbols_zs = set()
        for s1 in symbols_list:
            if s1 not in self._symbols:
                self._symbols.add(s1)
                self._symbols_zs.add(s1[:-3] + 'ZS')

        self._contract_size_dict = {}
        self._tick_size_dict = {}

        self._market_data = None
        self.result_fold = result_fold

    def _get_market_info(self):
        print('================', self._symbols)
        self._contract_size_dict = DataFactory.get_contract_size_dict(symbols=list(self._symbols),
                                                                      asset_type=ASSETTYPE_FUTURE)
        self._tick_size_dict = DataFactory.get_tick_size_dict(symbols=list(self._symbols), asset_type=ASSETTYPE_FUTURE)

    def _get_history(self, startDate, endDate):
        self._market_data = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq=self._freq,
                                                          symbols=list(self._symbols), start_date=startDate,
                                                          end_date=endDate)
    def _get_history_daily_zs(self, startDate, endDate):
        self._market_data_daily_zs = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=list(self._symbols_zs), start_date=startDate,
                                                                end_date=endDate)

    def _get_history_daily(self, startDate, endDate):

        self._market_data_daily = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=list(self._symbols), start_date=startDate,
                                                                end_date=endDate)

    def gen_signal(self, symbol, **kwargs):
        capital = kwargs['capital']
        contract_size_list = self._contract_size_dict[symbol]
        capital_intial = 10 * capital / contract_size_list / len(self._symbols)
        N1_lst = self._params['s_period'][symbol[:-4]]
        N2_lst = self._params['l_period'][symbol[:-4]]
        # N1_lst = self._params['s_period']
        # N2_lst = self._params['l_period']
        pos_df_list = []
        signal_df_lst = []
        for i in range(len(N1_lst)):
            N1 = N1_lst[i]
            N2 = N2_lst[i]
            data = self._format_data(symbol, N1, N2)
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

    def cap_vol_by_rolling(self, vol, targetvol):
        idxs = vol.index
        for idx in range(len(idxs)):
            curDate = idxs[idx]
            vol[curDate] = max(vol[curDate], targetvol)
        return vol

    @runing_time
    def _format_data(self, symbol, N1, N2):

        targetVol = self._params['targetVol']
        volLookback = self._params['volLookback']
        data_daily = self._market_data_daily[symbol][['trade_date', 'contract_id']].assign(symbol=symbol)
        zs_data = self._market_data_daily_zs[symbol[:-3] + 'ZS'][['high', 'close', 'open', 'low', 'volume', 'trade_date']].assign(symbol=symbol)
        idx = data_daily.index
        data_daily = data_daily.merge(zs_data, on=['trade_date', 'symbol'], how='left')
        data_daily.index = idx

        data_daily = data_daily[data_daily['volume'] > 0]
        data_daily = data_daily.assign(price_return=lambda df: df.close/df.close.shift(1))
        data_daily = data_daily\
                         .assign(HH_s=lambda df: talib.MAX(df.high.values, N1)) \
                         .assign(LL_s=lambda df: talib.MIN(df.low.values, N1)) \
                         .assign(HH_l=lambda df: talib.MAX(df.high.values, N2)) \
                         .assign(LL_l=lambda df: talib.MIN(df.low.values, N2)) \
                         .assign(ma_s=lambda df: (df.HH_s + df.LL_s) / 2)\
                         .assign(ma_l=lambda df: (df.HH_l + df.LL_l) / 2)\
                         .assign(ma_s1=lambda df: df.ma_s.shift(1))\
                         .assign(ma_l1=lambda df: df.ma_l.shift(1))\
                         .assign(ave_p=lambda df: (2*df.close + df.high + df.low)/4)\
                         .loc[:, ['contract_id', 'ma_s', 'ma_l', 'ma_s1', 'ma_l1', 'ave_p', 'high', 'low', 'close', 'open', 'trade_date', 'price_return']] \
                         .rename(columns={'high': 'high_d', 'low': 'low_d', 'open': 'open_d'}) \
                         .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x))) \
                         .assign(trade_date_1=lambda df: df.trade_date.shift(1)) \
                         .assign(close_d=lambda df: df.close.shift(1))\
            .assign(trade_date_next=lambda df: df.trade_date.shift(-1))
        if volLookback != 0:
            realizedVol = data_daily['price_return'].ewm(
                span=volLookback, ignore_na=True, adjust=False).std(bias=True) * (252 ** 0.5)
            # if symbol not in ['T_VOL', 'TF_VOL']:
            realizedVol = self.cap_vol_by_rolling(realizedVol, targetVol)
            riskScaler = targetVol/realizedVol
            data_daily['riskScaler'] = riskScaler
        else:
            data_daily['riskScaler'] = 1
        return data_daily

    def run_test(self, startDate, endDate, **kwargs):
        self._get_market_info()
        self._get_history_daily_zs(startDate=startDate, endDate=endDate)
        self._get_history_daily(startDate=startDate, endDate=endDate)

        capital = kwargs['capital']
        signal_dataframe = []
        for symbol in self._symbols:
            signal_dataframe.append(
                self.gen_signal(symbol=symbol, capital=capital))
        signal_dataframe = pd.concat(signal_dataframe)
        if self.result_fold is not None:
            signal_dataframe.to_csv(self.result_fold + '\\' + 'signal_dataframe.csv')
        return signal_dataframe