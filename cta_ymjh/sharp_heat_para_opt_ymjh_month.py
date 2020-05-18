# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 16:50
# 每两年调一次参数，每个月回看两年加几个月
# @Author  : zhangfang
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import json
import seaborn as sns
import pytz
import data_engine.global_variable as global_variable
import datetime


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


def get_opt_date_df(start_date,dropna = True):
    opt_date_list = []
    for yr in range(2000,2021):
        for mt in range(1,13):
            dt = datetime.datetime(yr,mt,1,15,0,0,0)
            dt2 = datetime.datetime(yr,mt,1,15,0,0,0).astimezone(pytz.timezone(global_variable.DEFAULT_TIMEZONE))
            if dt > datetime.datetime.now():
                continue
            elif dt2 < start_date + datetime.timedelta(days=365):
                continue
            opt_date_list.append(dt2)
    opt_date_series = pd.Series(opt_date_list,index=range(len(opt_date_list)))
    opt_date_df = pd.concat([opt_date_series,opt_date_series.shift(1),opt_date_series.shift(-1)],axis=1)
    opt_date_df.columns = ['opt_date','last_opt_date','next_opt_date']
    opt_date_df['next_opt_date'] = opt_date_df['next_opt_date'].fillna(datetime.datetime.now().astimezone(pytz.timezone(global_variable.DEFAULT_TIMEZONE)))
    if dropna:
        opt_date_df = opt_date_df.dropna(subset=['opt_date','last_opt_date'])
    return opt_date_df


if __name__ == "__main__":
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
    symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],  # 农产品
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],  # 化工
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],  # 金属
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],  # 黑色系
                    'Equity': ['IF', 'IH', 'IC'],  # 股指
                    'Bonds': ['T', 'TF'],  # 债券
                    'PreciousMetal': ['AG', 'AU']}  # 贵金属
    # time_list = [('2010-01-01', '2011-01-01'), ('2011-01-01', '2012-01-01'), ('2012-01-01', '2013-01-01'),
    #              ('2013-01-01', '2014-01-01'), ('2014-01-01', '2015-01-01'), ('2015-01-01', '2016-01-01'),
    #              ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01'), ('2018-01-01', '2019-01-01'),
    #              ('2019-01-01', '2020-01-01')]
    time_list = [('2010-01-01', '2012-01-01'), ('2012-01-01', '2014-01-01'), ('2014-01-01', '2016-01-01'),
                 ('2016-01-01', '2018-01-01'), ('2018-01-01', '2020-01-01')]
    chatPath = 'e://Strategy//ymjh//fig//'
    state_name = 'sharp'

    s_period_lst = [i for i in range(30, 2, -1)]
    l_period_lst = [i for i in range(11, 71, 3)]

    # s_period_lst = [i for i in range(29, 3, -1)]
    # l_period_lst = [i for i in range(14, 68, 3)]

    lst = []
    for clas in class_lst:
        for (s_date, e_date) in time_list:
            best_sharp = 0
            state = []
            harvest = []
            s_period_best = 6
            l_period_best = 40

            for i in range(1, len(s_period_lst)-1):
                s_period = s_period_lst[i]
                for j in range(1, len(l_period_lst)-1):
                    l_period = l_period_lst[j]
                    if s_period >= l_period:
                        continue
                    sharp_lst = []
                    for m in range(i-1, i+2):
                        temp_s_period = s_period_lst[m]
                        for n in range(j-1, j+2):
                            temp_l_period = l_period_lst[n]
                            if temp_s_period >= temp_l_period:
                                sharp_lst.append(-10)
                            else:
                                result_folder = 'e://Strategy//YMJH//better//resRepo_ymjh_%s_%s_%s' % (
                                clas, temp_s_period, temp_l_period)
                                daily_returns = pd.read_csv(result_folder + '//daily_returns.csv', header=None)
                                daily_returns.columns = ['trade_date', 'daily_return']
                                temp = daily_returns[
                                    (daily_returns['trade_date'] >= s_date) & (daily_returns['trade_date'] < e_date)]
                                try:
                                    sharp = np.mean(temp.daily_return) / np.std(temp.daily_return) * math.pow(252, 0.5)
                                except Exception as e:
                                    print(str(e))
                                    sharp = -10
                                sharp_lst.append(sharp)
                                if np.mean(sharp_lst) > best_sharp:
                                    best_sharp = np.mean(sharp_lst)
                                    s_period_best = s_period
                                    l_period_best = l_period
            row = []
            row.append(clas)
            row.append(s_date)
            row.append(e_date)
            row.append(s_period_best)
            row.append(l_period_best)
            lst.append(row)
    ret = pd.DataFrame(lst, columns=['class', 's_date', 'e_date', 's_period', 'l_period'])
    print(ret)
    ret.to_csv('e:/Strategy/YMJH/para/' + 'para_opt_history.csv')



