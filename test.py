# -*- encoding: utf-8 -*-
'''
@File    :   test.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/4/1 16:42   WangQiuyi      1.0         None
'''
from EcComparator import ECMWF_WRF_comparator
from netCDF4 import Dataset
import utils
import wrf_classes
import datetime as dt
from wrf import ALL_TIMES, CoordPair
from PrepScore import TS, POD, BIAS, MAR, FSC
import pickle
import os
import pandas as pd
import numpy as np
import plot
import PrepScore
import matplotlib as mpl
import EcComparator


mpl.rcParams["font.family"] = 'Arial'  # 默认字体类型
mpl.rcParams["font.style"] = 'italic'


def test():
    center_extent = [0, -1]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    fore_num = 25
    accumulate_time = 8
    thresholdlist = [0.1, 1, 2.5, 5, 10, 15, 20, 25, 30]

    skills = ["ACC", "TS", "POD", "FSC", "ETS", "FAR", "BIAS", "MAR", "HSS", "BSS", "MAE", "RMSE"]

    # FOR subplot
    TS_dir = r'F:\WRF\DOUBLE_V1\TS'
    TSsheet = pd.DataFrame(np.zeros(shape=(len(solution), len(thresholdlist))), index=solution, columns=thresholdlist)

    FSC_dir = r'F:\WRF\DOUBLE_V1\FSC'
    FSCsheet = pd.DataFrame(np.zeros(shape=(len(solution), len(thresholdlist))), index=solution, columns=thresholdlist)

    POD_dir = r'F:\WRF\DOUBLE_V1\POD'
    PODsheet = pd.DataFrame(np.zeros(shape=(len(solution), len(thresholdlist))), index=solution, columns=thresholdlist)

    MAR_dir = r'F:\WRF\DOUBLE_V1\MAR'
    MARsheet = pd.DataFrame(np.zeros(shape=(len(solution), len(thresholdlist))), index=solution, columns=thresholdlist)

    # for threshold plot
    single_dir = r'F:\WRF\DOUBLE_V1'
    single_sheet = pd.DataFrame(np.zeros(shape=(len(solution), len(thresholdlist))), index=solution,
                                columns=thresholdlist)

    # for skill in skills:
    # print("plotting " + skill + "\n")
    for threshold in thresholdlist:
        # TSsheet = pd.DataFrame(np.zeros(shape=(4, 25)), index=solution, columns=np.arange(0, 25))
        for i in solution:
            ECf = r'D:\acdemic\adaptor.mars.internal-1662534842.0755644-19262-13-55386ac4-6554-4a91-bdc6-f21aa74ae9d8.nc'
            # F:\WRF\Micheal_V1_2018101112_01h\CON0_01h
            WRFf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc'
            outf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrf_EC.pkl'
            stime = dt.datetime(2021, 8, 29, 00, 00, 00)

            wrf_config = {"dir": WRFf,
                          "stime": stime,
                          "file_mode": "single",
                          "step": 24,
                          "origininterval": 3,
                          "outputinterval": 3,
                          "element": "rain",
                          "units": "mm",
                          "timeidx": ALL_TIMES}
            wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
                                            step=wrf_config['step'],
                                            interval=wrf_config['outputinterval'])
            ECMWF_config = {"dir": ECf,
                            "file_mode": "single",  # dir类型
                            "element": "tp",
                            "units": "m",
                            "stime": wrf_time.stime,
                            "etime": wrf_time.etime}

            # ECMWF_WRF_comparator(ECMWF_config, wrf_config, outf)
            # data = Dataset(ECf, mode='r')
            # time = data['time']
            # EC_UTC = utils.ECMWF_to_UTC(time)

            EC_WRF_f = os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_WRF_in_same_grid.pkl')
            with open(EC_WRF_f, 'rb') as o:
                try:
                    data = pickle.load(o)
                except EOFError:
                    return None
            # 读取EC差值数据和WRF数据并进行比较
            # single_sheet.loc[i, threshold] = PrepScore.PRECIPITION_SKILL_SCORE(np.sum(data["EC"][0:accumulate_time, :, :], axis=0), np.sum(data["WRF"][0:accumulate_time, :, :], axis=0), mode=skill, threshold=threshold)

            # if not os.path.exists(os.path.join(single_dir, skill)):
            #     os.makedirs(os.path.join(single_dir, skill))
            # plot.PrepSkillDataFramePlot(single_sheet, skill, savedir=os.path.join(single_dir, skill))

            # single_dir = r'F:\WRF\DOUBLE_V1\fore_time_prep_score'
            # single_sheet = pd.DataFrame(np.zeros(shape=(len(solution), fore_num)), index=solution,
            #                             columns=np.arange(0, 25, 1))
            # # time ordered list
            # threshold = 0.1
            # for skill in skills:
            #     for timeindex in range(fore_num):
            #         # TSsheet = pd.DataFrame(np.zeros(shape=(4, 25)), index=solution, columns=np.arange(0, 25))
            #         for i in solution:
            #             ECf = r'D:\acdemic\adaptor.mars.internal-1662534842.0755644-19262-13-55386ac4-6554-4a91-bdc6-f21aa74ae9d8.nc'
            #             # F:\WRF\Micheal_V1_2018101112_01h\CON0_01h
            #             WRFf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc'
            #             outf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrf_EC.pkl'
            #             stime = dt.datetime(2021, 8, 29, 00, 00, 00)
            #
            #             wrf_config = {"dir": WRFf,
            #                           "stime": stime,
            #                           "file_mode": "single",
            #                           "step": 24,
            #                           "origininterval": 3,
            #                           "outputinterval": 3,
            #                           "element": "rain",
            #                           "units": "mm",
            #                           "timeidx": ALL_TIMES}
            #             wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
            #                                             step=wrf_config['step'],
            #                                             interval=wrf_config['outputinterval'])
            #             ECMWF_config = {"dir": ECf,
            #                             "file_mode": "single",  # dir类型
            #                             "element": "tp",
            #                             "units": "m",
            #                             "stime": wrf_time.stime,
            #                             "etime": wrf_time.etime}
            #
            #             # ECMWF_WRF_comparator(ECMWF_config, wrf_config, outf)
            #             # data = Dataset(ECf, mode='r')
            #             # time = data['time']
            #             # EC_UTC = utils.ECMWF_to_UTC(time)
            #
            #             EC_WRF_f = os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_WRF_in_same_grid.pkl')
            #             with open(EC_WRF_f, 'rb') as o:
            #                 try:
            #                     data = pickle.load(o)
            #                 except EOFError:
            #                     return None
            #             # 读取EC差值数据和WRF数据并进行比较
            #             single_sheet.loc[i, timeindex] = PrepScore.PRECIPITION_SKILL_SCORE(
            #                 data["EC"][timeindex, :, :], data["WRF"][timeindex, :, :],
            #                 mode=skill, threshold=threshold)
            #
            #     if not os.path.exists(os.path.join(single_dir, skill)):
            #         os.makedirs(os.path.join(single_dir, skill))
            #     single_sheet.to_csv(os.path.join(single_dir, skill, skill+".csv"))
            #     plot.PrepSkillDataFramePlot(single_sheet, skill, savedir=os.path.join(single_dir, skill))

            TSsheet.loc[i, threshold] = TS(np.sum(
                data["EC"][0:accumulate_time, center_extent[0]:center_extent[1], center_extent[0]:center_extent[1]],
                axis=0),
                                           np.sum(data["WRF"][0:12, center_extent[0]:center_extent[1],
                                                  center_extent[0]:center_extent[1]], axis=0), threshold=threshold)
            FSCsheet.loc[i, threshold] = FSC(np.sum(
                data["EC"][0:accumulate_time, center_extent[0]:center_extent[1], center_extent[0]:center_extent[1]],
                axis=0),
                                             np.sum(data["WRF"][0:12, center_extent[0]:center_extent[1],
                                                    center_extent[0]:center_extent[1]], axis=0), threshold=threshold)
            PODsheet.loc[i, threshold] = POD(np.sum(
                data["EC"][0:accumulate_time, center_extent[0]:center_extent[1], center_extent[0]:center_extent[1]],
                axis=0),
                                             np.sum(data["WRF"][0:accumulate_time, center_extent[0]:center_extent[1],
                                                    center_extent[0]:center_extent[1]], axis=0), threshold=threshold)
            MARsheet.loc[i, threshold] = MAR(np.sum(
                data["EC"][0:accumulate_time, center_extent[0]:center_extent[1], center_extent[0]:center_extent[1]],
                axis=0),
                                             np.sum(data["WRF"][0:accumulate_time, center_extent[0]:center_extent[1],
                                                    center_extent[0]:center_extent[1]], axis=0), threshold=threshold)

    TSsheet.to_csv(os.path.join(TS_dir, "TS_score" + ".csv"))
    FSCsheet.to_csv(os.path.join(FSC_dir, "FSC_score" + ".csv"))
    PODsheet.to_csv(os.path.join(POD_dir, "POD_score" + ".csv"))
    MARsheet.to_csv(os.path.join(MAR_dir, "MAR_score" + ".csv"))

    plot.prep_score_sub_plot(os.path.join(TS_dir, "TS_score" + ".csv"),
                             os.path.join(FSC_dir, "FSC_score" + ".csv"),
                             os.path.join(POD_dir, "POD_score" + ".csv"),
                             os.path.join(MAR_dir, "MAR_score" + ".csv"),
                             [0, 0.6], [0, 0.6], [0.1, 1.6], [0, 1], "threshold")
    # temp = PrepScore.PRECIPITION_SKILL_SCORE(np.sum(data["EC"][0:accumulate_time, :, :], axis=0), np.sum(data["WRF"][0:12, :, :], axis=0), mode="HSS", threshold=threshold)

    s = 1


def test_vert_cross():
    bestrack_lat = [26.7, 27.6, 28.5, 29.2,
                    29.9, 30.6, 31.5, 32.2,
                    33.0, 33.8, 34.4, 35.1, 35.8]
    bestrack_lon = [-87.6, -88.7, -89.6, -90.4,
                    -90.6, -90.8, -90.9, -90.5,
                    -90.0, -89.4, -88.4, -87.1, -85.5]
    WRF_track_dirs = [r"F:\WRF\DOUBLE_V1\CON0\CON0.csv",
                      r"F:\WRF\DOUBLE_V1\VAR1\VAR1.csv",
                      r"F:\WRF\DOUBLE_V1\VAR2\VAR2.csv",
                      r"F:\WRF\DOUBLE_V1\VAR3\VAR3.csv"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    wrf_fore_time = [6]
    for wrf_f in wrf_fore_time:
        for i, each_track in zip(solution, WRF_track_dirs):
            WRFf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc'
            wrf_track = pd.read_csv(each_track)
            wrf_track_lon, wrf_track_lat = wrf_track['lon'], wrf_track['lat']
            x_wind, y_wind = wrf_track_lon[wrf_f], wrf_track_lat[wrf_f]
            scale = 5
            start_point = CoordPair(lat=y_wind, lon=x_wind - scale)
            end_point = CoordPair(lat=y_wind, lon=x_wind + scale)
            # plot.vertical_cross_wspd_plot(WRFf, wrf_f, start_point, end_point, savedir=r'F:\WRF\DOUBLE_V1' + '\\' + i + r"\vert_wspd")
            plot.vertical_cross_temp_plot(WRFf, wrf_f, start_point, end_point,
                                          savedir=r'F:\WRF\DOUBLE_V1' + '\\' + i + r"\vert_temp")
    pass


def test_vert_cross_sub():
    WRF_track_dirs = [r"F:\WRF\DOUBLE_V1\CON0\CON0.csv",
                      r"F:\WRF\DOUBLE_V1\VAR1\VAR1.csv",
                      r"F:\WRF\DOUBLE_V1\VAR2\VAR2.csv",
                      r"F:\WRF\DOUBLE_V1\VAR3\VAR3.csv"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    wrf_fore_time = [0, 6, 12, 18]
    for wrf_f in wrf_fore_time:
        # for i, each_track in zip(solution, WRF_track_dirs):
        WRFf = [r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc' for i in solution]
        wrf_track = pd.read_csv(WRF_track_dirs[0])
        wrf_track_lon, wrf_track_lat = wrf_track['lon'], wrf_track['lat']
        x_wind, y_wind = wrf_track_lon[wrf_f], wrf_track_lat[wrf_f]
        scale = 5
        start_point = CoordPair(lat=y_wind, lon=x_wind - scale)
        end_point = CoordPair(lat=y_wind, lon=x_wind + scale)
        # plot.vertical_cross_wspd_plot(WRFf, wrf_f, start_point, end_point, savedir=r'F:\WRF\DOUBLE_V1' + '\\' + i + r"\vert_wspd")
        plot.vertical_cross_temp_plot(WRFf, wrf_f, start_point, end_point,
                                         savedir=r'F:\WRF\DOUBLE_V1' + '\\' + r"\vert_temp_sub")
    pass


def ctt_sub_plot():
    WRF_track_dirs = [r"F:\WRF\DOUBLE_V1\CON0\CON0.csv",
                      r"F:\WRF\DOUBLE_V1\VAR1\VAR1.csv",
                      r"F:\WRF\DOUBLE_V1\VAR2\VAR2.csv",
                      r"F:\WRF\DOUBLE_V1\VAR3\VAR3.csv"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    wrf_fore_time = [6]
    extent = [-75, -105, 10, 40]
    for wrf_f in wrf_fore_time:
        # for i, each_track in zip(solution, WRF_track_dirs):
        WRFf = [r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc' for i in solution]
        wrf_track = pd.read_csv(WRF_track_dirs[0])
        wrf_track_lon, wrf_track_lat = wrf_track['lon'], wrf_track['lat']
        x_wind, y_wind = wrf_track_lon[wrf_f], wrf_track_lat[wrf_f]
        scale = 5
        start_point = CoordPair(lat=y_wind, lon=x_wind - scale)
        end_point = CoordPair(lat=y_wind, lon=x_wind + scale)
        # plot.vertical_cross_wspd_plot(WRFf, wrf_f, start_point, end_point, savedir=r'F:\WRF\DOUBLE_V1' + '\\' + i + r"\vert_wspd")
        plot.ctt_subplot(WRFf, extent, wrf_f, start_point, end_point,
                         savedir=r'F:\WRF\DOUBLE_V1' + '\\' + r"\ctt_sub")
    pass


def test_cct_verttemp_subplot():
    WRF_track_dirs = [r"F:\WRF\DOUBLE_V1\CON0\CON0.csv",
                      r"F:\WRF\DOUBLE_V1\VAR1\VAR1.csv",
                      r"F:\WRF\DOUBLE_V1\VAR2\VAR2.csv",
                      r"F:\WRF\DOUBLE_V1\VAR3\VAR3.csv"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    wrf_fore_time = [0, 4, 6, 8, 10]
    extent = [-75, -105, 10, 40]
    for wrf_f in wrf_fore_time:
        # for i, each_track in zip(solution, WRF_track_dirs):
        WRFf = [r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc' for i in solution]
        wrf_track = pd.read_csv(WRF_track_dirs[0])
        wrf_track_lon, wrf_track_lat = wrf_track['lon'], wrf_track['lat']
        x_wind, y_wind = wrf_track_lon[wrf_f], wrf_track_lat[wrf_f]
        scale = 5
        start_point = CoordPair(lat=y_wind, lon=x_wind - scale)
        end_point = CoordPair(lat=y_wind, lon=x_wind + scale)
        # plot.vertical_cross_wspd_plot(WRFf, wrf_f, start_point, end_point, savedir=r'F:\WRF\DOUBLE_V1' + '\\' + i + r"\vert_wspd")
        plot.ctt_vert_wspd_temp_subplot(WRFf, extent, wrf_f, start_point, end_point,
                         savedir=r'F:\WRF\DOUBLE_V1' + '\\' + r"\ctt_vertwspdtemp_subplot")


def test_ERA5_gird():
    fera5 = r"F:\WRF\DOUBLE_V1\EC\pwv_z\adaptor.mars.internal-20210829-3days.grib"
    fNorthAmSiteCoor = r"D:\acdemic\毕业论文\PWVDATA\NorthAmericanSiteCoordinate.txt"
    SiteCoor = pd.read_csv(fNorthAmSiteCoor, sep="\s+")
    EcComparator.ECMWF_GPSPWV_comparator(fera5, SiteCoor)


# test()
# test_vert_cross_sub()
# ctt_sub_plot()
# test_cct_verttemp_subplot()
test_ERA5_gird()
