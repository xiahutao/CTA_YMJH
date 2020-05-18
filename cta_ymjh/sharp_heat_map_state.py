# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 16:50
# @Author  : zhangfang
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import json
import seaborn as sns
# import tensorflow as tf
# tf.__version__
def makeHeatMap(path, chatPath, pair, dirs, aft):
    """"""
    pnl_list = list()
    for d in dirs:
        if '_' + pair in d:
            speed = json.loads(d.split('_speeds_')[1])[0]
            pnl_df = pd.read_csv(path + '/' + d + f'/daily_pnl{aft}.csv', index_col=0)
            pnl_df.index = pd.to_datetime(pnl_df.index).tz_convert('PRC')
            pnl_df['year'] = pnl_df.index.year
            pnl_year = pnl_df.groupby(by=['year'])[f'daily_pnl{aft}'].sum()
            pnl_year.name = speed
            pnl_year = pd.DataFrame(pnl_year)
            pnl_list.append(pnl_year)
            del pnl_df
    pnl_data = pd.concat(pnl_list, axis=1)
    pnl_data.sort_index(axis=1, ascending=False, inplace=True)
    pnl_data = pnl_data / 10000000
    del pnl_list
    f, ax1 = plt.subplots(figsize=(len(pnl_data), len(pnl_data.columns)))
    # pnl_data = pnl_data.corr()  # pt为数据框或者是协方差矩阵
    vmax = np.abs(pnl_data).max().max()
    vmin = -vmax
    sns.heatmap(pnl_data.T, annot=True, ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'weight': 'bold', 'color': 'blue'},
                cmap='rainbow')
    ax1.set_title(pair + aft)
    ax1.set_ylabel('speed')
    plt.savefig(chatPath + f'/{pair}{aft}_ThermodynamicChart.png')
    print(f'/{pair}{aft}_ThermodynamicChart.png', '完成')


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
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
    symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],  # 农产品
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],  # 化工
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],  # 金属
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],  # 黑色系
                    'Equity': ['IF', 'IH', 'IC'],  # 股指
                    'Bonds': ['T', 'TF'],  # 债券
                    'PreciousMetal': ['AG', 'AU'],  # 贵金属
                    'SoftComm': ['CF', 'CS', 'SR']}  # 软商品
    class_lst = ['SoftComm']
    time_list = [('2010-01-01', '2012-01-01'), ('2012-01-01', '2014-01-01'), ('2014-01-01', '2016-01-01'),
                 ('2016-01-01', '2018-01-01'), ('2018-01-01', '2020-01-01')]
    chatPath = 'e://Strategy//ymjh//fig//'
    state_name = 'sharp'

    s_period_lst = [i for i in range(30, 2, -1)]
    l_period_lst = [i for i in range(11, 71, 3)]

    for clas in class_lst:
        for (s_date, e_date) in time_list:
            state = []
            harvest = []

            for s_period in s_period_lst:
                harvest_row = []
                for l_period in l_period_lst:
                    if s_period >= l_period:
                        harvest_row.append(0)
                    else:
                        result_folder = 'e://Strategy//YMJH//better//resRepo_ymjh_%s_%s_%s' % (clas, s_period, l_period)
                        daily_returns = pd.read_csv(result_folder + '//daily_returns.csv', header=None)
                        daily_returns.columns = ['trade_date', 'daily_return']
                        print(len(daily_returns))
                        temp = daily_returns[(daily_returns['trade_date'] >= s_date) & (daily_returns['trade_date'] < e_date)]
                        print(len(temp))
                        try:
                            sharp = np.mean(temp.daily_return)/np.std(temp.daily_return) * math.pow(252, 0.5)
                        except Exception as e:
                            print(str(e))
                            sharp = -1
                        harvest_row.append(sharp)
                harvest.append(harvest_row)
            x_label = s_period_lst
            y_label = l_period_lst
            print(harvest)
            harvest = np.array(harvest)
            print(harvest)

            fig, ax1 = plt.subplots(figsize=(len(x_label), len(y_label)), nrows=1)
            # fig, ax1 = plt.subplots()

            vmax = max(max(harvest[i]) for i in range(len(harvest)))
            print(vmax)
            vmin = -vmax
            h = sns.heatmap(harvest, annot=True, fmt='.2f', ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'size': 20}, cbar=False)
            cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
            cb.ax.tick_params(labelsize=28)
            ax1.set_title(s_date + '_' + state_name + ' of ' + clas, fontsize=28)
            ax1.set_xticklabels(l_period_lst, fontsize=20)
            ax1.set_yticklabels(s_period_lst, fontsize=20)
            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

            ax1.set_xlabel('l_period', fontsize=24)
            ax1.set_ylabel('s_period', fontsize=24)
            fig.tight_layout()
            plt.savefig(chatPath + state_name + '//' + s_date + '_' + state_name + '_' + clas + '.png')
            plt.show()
            # 参数组合下的sharpe的分布图

    #
    #
    # sharp_lst = []
    # for clas in class_lst:
    #     harvest = []
    #     index_dict_list = []
    #     for s_period in s_period_lst:
    #         harvest_row = []
    #         for l_period in l_period_lst:
    #             if l_period != 2 * s_period:
    #                 continue
    #             # l_period = 2 * s_period
    #             result_folder = 'e://Strategy//MOMENTUM//better//resRepo_momentum_%s_%s_%s' % (clas, s_period, l_period)
    #             daily_return_list = pd.read_csv(result_folder + '//daily_returns.csv', index_col=0,
    #                                             header=None).iloc[:, 0]
    #             net_lst = (1 + daily_return_list).cumprod()
    #             sharp = yearsharpRatio(net_lst, 1)
    #             print(sharp)
    #             sharp_lst.append(sharp)
    #             harvest_row.append(sharp)
    #         harvest.append(harvest_row)
    #     x_label = s_period_lst
    #     y_label = l_period_lst
    #     print(harvest)
    #     harvest = np.array(harvest)
    #     print(harvest)
    #
    #     fig, ax1 = plt.subplots(figsize=(len(x_label), len(y_label)), nrows=1)
    #     # fig, ax1 = plt.subplots()
    #
    #     vmax = max(max(harvest[i]) for i in range(len(harvest)))
    #     print(vmax)
    #     vmin = -vmax
    #     h = sns.heatmap(harvest, annot=True, fmt='.2f', ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'size': 20}, cbar=False)
    #     cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    #     cb.ax.tick_params(labelsize=28)
    #     ax1.set_title(state_name + ' of ' + clas, fontsize=28)
    #     ax1.set_xticklabels(l_period_lst, fontsize=20)
    #     ax1.set_yticklabels(s_period_lst, fontsize=20)
    #     font = {'family': 'serif',
    #             'color': 'darkred',
    #             'weight': 'normal',
    #             'size': 16,
    #             }
    #
    #     ax1.set_xlabel('l_period', fontsize=24)
    #     ax1.set_ylabel('s_period', fontsize=24)
    #     fig.tight_layout()
    #     plt.savefig(chatPath + state_name + '//' + state_name + ' of ' + clas + '.png')
    #     plt.show()
    #     fig = plt.figure(figsize=(9, 9), facecolor='gray')
    #     plt.hist(sharp_lst, color="g", histtype="bar", rwidth=1, alpha=0.6)
    #     plt.title('hist_sharpe_' + clas, fontsize=28)
    #     plt.savefig(chatPath + state_name + '//' + 'hist_sharpe_' + clas + '.png')
    #     plt.show()
