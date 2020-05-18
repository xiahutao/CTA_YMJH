# -*- coding: utf-8 -*-
import sys
import os
CurrentPath = os.path.dirname(__file__)
print(CurrentPath)
sys.path.append(CurrentPath.replace('cta_momentum', ''))
print(sys.path)
import datetime
import warnings
import copy
warnings.filterwarnings("ignore")
import traceback
import math
import numpy as np
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
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
# DataFactory.sync_future_from_remote()


def maxRetrace(lst, n):
    '''
    :param list:netlist
    :param n:每交易日对应周期数
    :return: 最大历史回撤
    '''
    Max = 0
    new_lst = copy.deepcopy(lst)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]

    for i in range(len(new_lst)):
        if 1 - new_lst[i] / max(new_lst[:i + 1]) > Max:
            Max = 1 - new_lst[i] / max(new_lst[:i + 1])
    return Max


def annROR(netlist, n):
    '''
    :param netlist:净值曲线
    :param n:每交易日对应周期数
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 252 * n / len(netlist)) - 1


def daysharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row)


def yearsharpRatio(netlist, n):
    '''
    :param netlist:
    :param n: 每交易日对应周期数
    :return:
    '''
    row = []
    new_lst = copy.deepcopy(netlist)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]
    for i in range(1, len(new_lst)):
        row.append(math.log(new_lst[i] / new_lst[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


if __name__ == "__main__":
    symbols = ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM', 'IF', 'IH',
                   'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']  # 所有品种
    symbols = [i + '_VOL' for i in symbols]
    result_folder = 'e://Strategy//'
    n = 1
    period = 1
    volLookback = 0
    s_date_lst = [str(i) for i in range(2012, 2021, 1)]
    # lst = []
    # for each in symbols:
    #     run_symbols = [each]
    #
    #     sharp = pd.read_csv(result_folder + '%s_mutipara' % ('_'.join([i[:-4] for i in run_symbols])) + '//sharpe_dataframe.csv')
    #     lst.append(sharp)
    # ret = pd.concat(lst)
    # ret.to_csv(result_folder + '//sharp_all.csv')
    daily_return1 = pd.read_csv('E://Strategy//MOMENTUM//resRepo_momentum_open_exec_J_HC_RB_I_NI_TF_SM_AL_RU_MA_SR_P_TA_T_SC_IF_Y_FU_IH_AG_PB//daily_returns.csv',
                               header=None)
    daily_return1.columns = ['date', 'chng1']
    daily_return2 = pd.read_csv('E://Strategy//YMJH//backtest//2006AP_HC_J_TA_SC_I_RU_TF_RB_ZC_M_MA_AU_AG_NI_C_CU_V_BU_SF_PB_A_T_mutipara_zs//daily_returns.csv',
                               header=None)
    daily_return2.columns = ['date', 'chng2']
    daily_return3 = pd.read_csv(
        'E://Strategy//TCS//backtest//2008_mutipara_SC_J_TA_I_MA_RU_ZC_P_SF_CF_NI_TF_IF_SM_PB_T_SR_AL_BU_AU_AP_FU_C//daily_returns.csv',
        header=None)
    daily_return3.columns = ['date', 'chng3']
    daily_return = daily_return1.merge(daily_return2, on=['date']).merge(daily_return3, on=['date'])
    daily_return['chng'] = (daily_return['chng1'] + daily_return['chng2'] + daily_return['chng3']) / 3
    daily_return['net'] = (daily_return['chng'] + 1).cumprod()
    daily_return['date'] = daily_return['date'].apply(lambda x: str(x)[:10])
    print(daily_return)
    date_lst = [('2010-01-01', '2011-01-01'), ('2011-01-01', '2012-01-01'), ('2012-01-01', '2013-01-01'),
                ('2013-01-01', '2014-01-01'), ('2014-01-01', '2015-01-01'), ('2015-01-01', '2016-01-01'),
                ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01'), ('2018-01-01', '2019-01-01'),
                ('2019-01-01', '2020-01-01'), ('2010-01-01', '2020-01-01'), ('2010-01-01', '2020-01-01'),
                ('2015-01-01', '2020-01-01'), ('2017-01-01', '2020-01-01'), ('2020-01-01', '2020-10-01')]
    lst = []
    for (s_date, e_date) in date_lst:
        df = daily_return[(daily_return['date'] >= s_date) & (daily_return['date'] < e_date)]
        net_lst_ = df.net.tolist()
        ann_ROR = annROR(net_lst_, n)
        max_retrace = maxRetrace(net_lst_, n)
        sharp = yearsharpRatio(net_lst_, n)
        row = []
        row.append(s_date)
        row.append(e_date)
        row.append(sharp)
        row.append(ann_ROR)
        row.append(max_retrace)
        lst.append(row)
        df['net'] = df['net']/df['net'].tolist()[0]
        df[['date', 'net']].plot(
            x='date', kind='line', grid=True,
            title=s_date + '_' + e_date)
        plt.show()
    ret = pd.DataFrame(lst, columns=['s_date', 'e_date', 'sharp', 'ann_ROR', 'max_retrace'])
    # ret.to_csv(result_folder + '//sharp_diff_period.csv')


