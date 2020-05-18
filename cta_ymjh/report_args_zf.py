import os
import pandas
import numpy
from data_engine.data_factory import DataFactory
from data_engine.instrument.future import Future
# from analysis.report.file_tools import file_tools
from itertools import combinations
from analysis.sector_analysis import SectorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    DataFactory.config('jz2018*')

    # path = r'E:\PairStrategy_args_del'
    # arg_list = ['volLookback', 'scaler', 'speeds']
    # filename = 'daily_return_by_init_aum.csv'
    #
    # # 遍历提取回测文件名
    # dr_filename_dict = file_tools.get_pathfile_list(path=path, filename=filename)
    # dr_filename_dict_1 = {x: y.split('\\') + [y] for x, y in dr_filename_dict.items()}
    #
    # # 加工得到参数与文件名的dataframe,  columns指定 dataframe对应的列头
    # columns = ['root', 'folder1'] + arg_list + ['pair', 'filename', 'pathfile']
    # args_pathfile_df = pandas.DataFrame(dr_filename_dict_1.values(), columns=columns)[
    #     arg_list + ['pair', 'filename', 'pathfile']]
    # args_pathfile_df['sector'] = None
    # future_dict = {}
    # for idx in range(len(args_pathfile_df)):
    #     row = args_pathfile_df.iloc[idx]
    #     pair = row['pair']
    #     first_future = pair.split('_')[0]
    #     if first_future in future_dict:
    #         first_future_obj = future_dict[first_future]
    #     else:
    #         first_future_obj = Future(first_future)
    #         future_dict[first_future] = first_future_obj
    #     args_pathfile_df.loc[row.name, 'sector'] = first_future_obj.sector
    #
    # # 加载daily_return
    # pathfile_daily_return_dict = {row['pathfile']: pandas.read_csv(row['pathfile'], header=None,
    #                                                                names=['date_index', row['pair']]).set_index(
    #     'date_index')[row['pair']]
    #                               for idx, row in args_pathfile_df.iterrows()}

    # 遍历参数对，给出热力图
    symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
                    'Equity': ['IF', 'IH', 'IC'],
                    'Bonds': ['T', 'TF'],
                    'PreciousMetal': ['AG', 'AU']}
    s_period_lst = [i for i in range(3, 31)]
    l_period_lst = [i for i in range(10, 71, 1)]
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
    for clas in class_lst:
        index_dict_list = []
        for s_period in s_period_lst:
            for l_period in l_period_lst:
                result_folder = 'e://Strategy//YMJH//better//resRepo_ymjh_%s_%s_%s' % (clas, s_period, l_period)
                daily_return_list = pd.read_csv(result_folder + '//daily_returns.csv', index_col=0, header=None).iloc[:, 0]
                print(daily_return_list)
                if len(daily_return_list) > 0:
                    sectoranalysis = SectorAnalysis(sector=clas, daily_returns_list=daily_return_list)
                    index_dict = sectoranalysis.get_index_dict()
                    index_dict['s_period'] = s_period
                    index_dict['l_period'] = l_period
                    index_dict_list.append(index_dict)
                if len(index_dict_list) > 0:
                    df = pandas.DataFrame([pandas.Series(x) for x in index_dict_list])
                    df_pivot = df.pivot_table(values='sharpe', index='s_period', columns='l_period')
                    print(df_pivot)
                    sns.heatmap(
                        df_pivot.fillna(0) * 100.0,
                        annot=True,
                        annot_kws={"size": 9},
                        alpha=1.0,
                        center=0.0,
                        cbar=False)
                    plt.savefig('heatmap_' + clas + '.png')

    # arg_pairs = [x for x in combinations(arg_list, 2)]
    # for each_arg_pair in arg_pairs:
    #     index_dict_list = []
    #     for x, sub_df in args_pathfile_df.groupby(list(each_arg_pair)):
    #         pathfile_list = sub_df['pathfile'].unique()
    #         daily_return_list = [y for x, y in pathfile_daily_return_dict.items() if x in pathfile_list]
    #         if len(daily_return_list) > 0:
    #             sectoranalysis = SectorAnalysis(sector='_'.join(x), daily_returns_list=daily_return_list)
    #             index_dict = sectoranalysis.get_index_dict()
    #             index_dict[each_arg_pair[0]] = x[0]
    #             index_dict[each_arg_pair[1]] = x[1]
    #             index_dict_list.append(index_dict)
    #     if len(index_dict_list) > 0:
    #         df = pandas.DataFrame([pandas.Series(x) for x in index_dict_list])
    #         df_pivot = df.pivot_table(values='sharpe', index=each_arg_pair[0], columns=each_arg_pair[1])
    #         print(df_pivot)
    #         sns.heatmap(
    #             df_pivot.fillna(0) * 100.0,
    #             annot=True,
    #             annot_kws={"size": 9},
    #             alpha=1.0,
    #             center=0.0,
    #             cbar=False)
    #         plt.savefig('heatmap_' + '_'.join(each_arg_pair) + '.png')
    #
    # # 参数组合下的sharpe的分布图
    # index_dict_list = []
    # for x, sub_df in args_pathfile_df.groupby(arg_list):
    #     pathfile_list = sub_df['pathfile'].unique()
    #     daily_return_list = [y for x, y in pathfile_daily_return_dict.items() if x in pathfile_list]
    #     if len(daily_return_list) > 0:
    #         sectoranalysis = SectorAnalysis(sector='_'.join(x), daily_returns_list=daily_return_list)
    #         index_dict = sectoranalysis.get_index_dict()
    #         index_dict[each_arg_pair[0]] = x[0]
    #         index_dict[each_arg_pair[1]] = x[1]
    #         index_dict_list.append(index_dict)
    #
    # sharpe_list = [x['sharpe'] for x in index_dict_list]
    # fig = plt.figure(figsize=(9, 9), facecolor='gray')
    # plt.hist(sharpe_list, color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.savefig('hist_sharpe.png')
    # print(args_pathfile_df)
