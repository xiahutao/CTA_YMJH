# -*- coding: utf-8 -*-
import sys
import os
CurrentPath = os.path.dirname(__file__)
# print(CurrentPath)
sys.path.append(CurrentPath.replace('residula_mom',''))
# print(sys.path)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import re
import json


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
    # plt.show()


if __name__ == '__main__':
    path = 'G:/PairStrategy'
    freq = '1M'
    aft = '_fee'  # _gross
    chatPath = path + '/thermodynamic_chart' + aft
    try:
        os.mkdir(chatPath)
    except:
        pass

    dirs = os.listdir(path)
    pairs_set = set()
    for d in dirs:
        try:
            pair = (re.search(r'[A-Z]+_VOL_[A-Z]+_VOL_{}'.format(freq), d)).group()
        except:
            continue
        pairs_set.add(pair)
    chatFiles = os.listdir(chatPath)
    fails = list()
    for pair in pairs_set:
        if f'{pair}{aft}_ThermodynamicChart.png' in chatFiles:
            continue
        try:
            makeHeatMap(path, chatPath, pair, dirs, aft)
        except Exception as e:
            fails.append(pair)
    print(fails)
