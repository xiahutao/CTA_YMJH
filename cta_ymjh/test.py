# -*- coding: utf-8 -*-
import sys
import os
CurrentPath = os.path.dirname(__file__)
print(CurrentPath)
sys.path.append(CurrentPath.replace('cta_momentum', ''))
print(sys.path)
import datetime
import warnings
import time
warnings.filterwarnings("ignore")
import traceback
import pandas as pd
from execution.execution import Execution
from analysis.analysis import Analysis
from cta_ymjh.ctaymjh import CtaYmjhStrategy
from settlement.settlement import Settlement
from data_engine.data_factory import DataFactory
import data_engine.setting as setting

from common.file_saver import file_saver
from common.os_func import check_fold
from data_engine.setting import ASSETTYPE_FUTURE, FREQ_1M, FREQ_5M, FREQ_1D
# DataFactory.sync_future_from_remote()


if __name__ == "__main__":
    symbols = ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM', 'IF', 'IH',
                   'IC', 'T', 'TF', 'AG', 'AU']  # 所有品种
    # symbols = ['CY']
    symbols = [i + '_VOL' for i in symbols]
    result_folder = 'e://Strategy//YMJH//backtest//%s_mutipara' % (
        '_'.join([i[:-4] for i in run_symbols]))
    exec_lag = 1
    period = 1
    volLookback = 0
    sharp_lst = []
    ret_path = 'E://Strategy//YMJH//backtest//daily_return_10para_opt'
    check_fold(ret_path)
    s_date_lst = [str(i) for i in range(2012, 2021, 1)]
    for each in symbols:
        run_symbols = [each]
        daily_return_lst = []
        for s_date in s_date_lst:
            # result_filename = 'opt_backtest_dailyreturns_' + '_'.join(run_symbols) + '.csv'
            result_filename = s_date + '_para10_opt_backtest_dailyreturns_' + '_'.join(run_symbols) + '.csv'

            try:
                daily_return = pd.read_csv('E://Strategy//YMJH//backtest//daily_return//' + result_filename, index_col=0, header=None)
                a = 0
                daily_return = daily_return[(daily_return.index >= s_date + '-01-01') & (daily_return.index < str(int(s_date) + 1) + '-01-01')]
                # daily_return = daily_return[(daily_return.index >= '2012' + '-01-01')]
                print(daily_return)
                daily_return_lst.append(daily_return)
                # daily_return.to_csv('E://Strategy//YMJH//backtest//daily_return_5para//' + result_filename, header=None)

            except Exception as e:
                print(e)
        ret = pd.concat(daily_return_lst)
        # ret.to_csv(ret_path + '//backtest_dailyreturns_' + '_'.join(run_symbols) + '.csv', header=None)
        ret.to_csv(ret_path + '//backtest_dailyreturns_' + '_'.join(run_symbols) + '.csv', header=None)


    # sharp_df = pd.concat(sharp_lst).assign(volLookback=volLookback)
    # sharp_df.to_csv('e://Strategy//YMJH//YMJH_args_opt//opt//sharp_all_opt' + '.csv')
