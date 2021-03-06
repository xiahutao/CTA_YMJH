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
from analysis.report.script.report_strategy_summary import report as report_strategy_summary
from analysis.report.script.report_strategy_sharpelist import report as report_strategy_sharpelist
warnings.filterwarnings("ignore")
import traceback
import pandas
from execution.execution import Execution
from analysis.analysis import Analysis
from cta_ymjh.ctaymjh_test import CtaYmjhStrategy
from settlement.settlement import Settlement
from data_engine.data_factory import DataFactory
import data_engine.setting as setting
from common.file_saver import file_saver
from common.os_func import check_fold
from data_engine.setting import ASSETTYPE_FUTURE, FREQ_1M, FREQ_5M, FREQ_1D


# DataFactory.sync_future_from_remote()


def run_backtest(s_date, run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, saving_file=False):
    import time
    t1 = time.clock()
    try:
        DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
        # 策略对象
        strategy_obj = CtaYmjhStrategy(symbols_list=run_symbols,
                                       freq=freq,
                                       asset_type=ASSETTYPE_FUTURE,
                                       result_fold=result_folder,
                                       **strategy_params
                                       )
        # 回测
        signal_dataframe = strategy_obj.run_test(startDate=run_params['start_date'], endDate=run_params['end_date'],
                                                 **run_params
                                                 )

        execution_obj = Execution(freq=freq, exec_price_mode=Execution.EXEC_BY_OPEN, exec_lag=exec_lag)
        (success, positions_dataframe) = execution_obj.exec_trading(signal_dataframe=signal_dataframe)

        if not success:
            print(positions_dataframe)
            assert False

        if success:
            settlement_obj = Settlement(init_aum=run_params['capital'])
            # file_saver().save_file(positions_dataframe, os.path.join(result_folder, 'positions_dataframe.csv'))
            settlement_obj.settle(positions_dataframe=positions_dataframe)
            print(settlement_obj.daily_return)
            result_filename = s_date + '_para10_opt_backtest_dailyreturns_' + '_'.join(run_symbols) + '.csv'
            check_fold('E://Strategy//YMJH//backtest//daily_return')
            settlement_obj.daily_return.to_csv('E://Strategy//YMJH//backtest//daily_return//' + result_filename)

            # 分析引擎，  结果保存到result_folder文件夹下
            analysis_obj = Analysis(daily_returns=settlement_obj.daily_return,
                                    daily_positions=settlement_obj.daily_positions,
                                    daily_pnl=settlement_obj.daily_pnl,
                                    daily_pnl_gross=settlement_obj.daily_pnl_gross,
                                    daily_pnl_fee=settlement_obj.daily_pnl_fee,
                                    transactions=settlement_obj.transactions,
                                    round_trips=settlement_obj.round_trips,
                                    result_folder=result_folder,
                                    strategy_id='_'.join(
                                        [strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]),
                                    symbols=strategy_obj._symbols,
                                    strategy_type=strategy_obj._strategy_name)
            sharpe_ratio = analysis_obj.sharpe_ratio()
            # sharpe_dataframe = pandas.DataFrame({'symbol': ['_'.join(run_symbols)], 'sharp': [sharpe_ratio]})
            # file_saver().save_file(sharpe_dataframe, os.path.join(result_folder, 'sharpe_dataframe.csv'))
            analysis_obj.plot_cumsum_pnl(show=False,
                                         title='_'.join([strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]))
            analysis_obj.plot_all()
            analysis_obj.save_result()
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    print('=================', 'run_backtest', '%.6fs' % (time.clock() - t1))


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_LOCAL)
    client = DataFactory.get_mongo_client()
    print(client.database_names())
    from multiprocessing import Pool, cpu_count

    pool = Pool(max(1, cpu_count() - 10))
    pool = Pool(4)
    method = '_para10_opt'

    # file_save_obj = file_saver()
    # symbols_all = ['AG', 'AL', 'AP', 'AU', 'B', 'C', 'CF', 'CU',
    #            'HC', 'I', 'IF', 'J', 'JD', 'JM', 'M', 'MA', 'NI',
    #            'PB', 'PP', 'RB', 'RU', 'SC', 'SM', 'SN', 'SR',
    #            'T', 'TA', 'TF', 'ZC']  # 所有品种
    # symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],
    #                 'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
    #                 'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
    #                 'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
    #                 'Equity': ['IF', 'IH', 'IC'],
    #                 'Bonds': ['T', 'TF'],
    #                 'PreciousMetal': ['AG', 'AU']}
    # class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
    # # class_lst = ['PreciousMetal']
    # s_date_lst = [str(i) for i in range(2012, 2021, 1)]
    # para_df_all = pandas.read_csv('e:/Strategy/YMJH/para_all.csv')
    # para_df_all.loc[:, ['s_date', 'e_date', 'class']] = para_df_all.loc[:, ['s_date', 'e_date', 'class']].astype(str)
    # para_df_all.loc[:, ['s_period', 'l_period']] = para_df_all.loc[:, ['s_period', 'l_period']].astype(int)
    # for s_date1 in s_date_lst:
    #     print(s_date1)
    #     para_df_ = para_df_all[para_df_all['s_date'] != s_date1]
    #     # para_df_ = para_df_all
    #     for clas in class_lst:
    #         if len(para_df_[para_df_['class'] == clas]) == 0:
    #             continue
    #         para_df = para_df_.set_index(['class'])
    #         symbols = symbols_dict[clas]
    #         symbols = [i + '_VOL' for i in symbols]
    #         run_symbols_0 = symbols
    #         for each in run_symbols_0:
    #             run_symbols = [each]
    #             run_params = {'capital': 400000000,
    #                           'daily_start_time': '9:00:00',
    #                           'daily_end_time': '23:30:00',
    #                           'start_date': '20100101',
    #                           'end_date': '20200306'
    #                           }
    #             try:
    #                 s_period_dict = {clas: para_df.loc[clas].s_period.values.tolist()}
    #                 l_period_dict = {clas: para_df.loc[clas].l_period.values.tolist()}
    #             except Exception as e:
    #                 s_period_dict = {clas: [para_df.loc[clas].s_period]}
    #                 l_period_dict = {clas: [para_df.loc[clas].l_period]}
    #             print(clas, s_period_dict[clas], l_period_dict[clas])
    #             strategy_params = {'period': 1440,
    #                                's_period': s_period_dict[clas],
    #                                'l_period': l_period_dict[clas],
    #                                'targetVol': 0.1,
    #                                'volLookback': 20
    #                                }
    #             if method == 'ori':
    #                 strategy_params = {'period': 1440,
    #                                    's_period': [6],
    #                                    'l_period': [40],
    #                                    'targetVol': 0.1,
    #                                    'volLookback': 20
    #                                    }
    #             exec_lag = 1
    #
    #             result_folder = 'e://Strategy//YMJH//backtest//%s_%s_mutipara' % (
    #                 clas, '_'.join(run_symbols))
    #             check_fold(result_folder)
    #             freq = FREQ_1M
    #             if strategy_params['period'] == 5:
    #                 freq = FREQ_5M
    #             elif strategy_params['period'] == 1:
    #                 freq = FREQ_1M
    #             else:
    #                 freq = FREQ_1D
    #             try:
    #                 # 策略对象
    #                 print('symbols', '_'.join(run_symbols))
    #                 pool.apply_async(run_backtest,
    #                                  args=(s_date1, run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, True))
    #                 # run_backtest(run_pairs, freq, result_folder, strategy_params, run_params, exec_lag,saving_file=False)
    #                 DataFactory().clear_data()
    #             except:
    #                 DataFactory().clear_data()
    #                 traceback.print_exc()
    # pool.close()
    # file_saver().join()
    # pool.join()

    backtest_path = r'E:/Strategy/YMJH/backtest'
    report_path = os.path.join(backtest_path, r'report')
    check_fold(report_path)

    opt_result_path = 'E://Strategy//YMJH//backtest//daily_return_10para_opt' # 各品种优化参数的回测结果文件夹
    format_folder_func = lambda x: x.replace('backtest_dailyreturns_', '').replace('.csv', '').split('_')[
        0]  # 从文件名提取品种标的， 可以实例化Future的代码
    strategy_name = 'resFuturesCtaYmjh' + method
    filename = None  # 不过滤文件，全部提取
    report_strategy_summary(strategy_name=strategy_name, format_folder_func=format_folder_func, path=opt_result_path,
                            filename=filename, pdf_path=report_path)
    report_strategy_sharpelist(strategy_name=strategy_name, format_folder_func=format_folder_func, path=opt_result_path,
                               filename=filename, pdf_path=report_path)