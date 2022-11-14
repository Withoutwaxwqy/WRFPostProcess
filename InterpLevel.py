# -*- encoding: utf-8 -*-
'''
@File    :   InterpLevel.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/11/1 16:47   WangQiuyi      1.0         None
'''

import wrf
import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np
import utils
import datetime as dt
import pickle
import plot


def interp_on_pressure_level(wrf_dir, element, units, timeidx, pressure_levels):
    """

    :param units:
    :param wrf_dir:
    :param element:
    :param pressure_levels:
    :return:
    """
    # 读数据
    wrf_in = Dataset(wrf_dir)
    pressure = getvar(wrf_in, "pressure")
    if element == 'rh':
        data = getvar(wrf_in, element, timeidx=timeidx)
    if element == 'z':
        data = getvar(wrf_in, element, units='m', timeidx=timeidx)[:]*9.80
    if element == 'geopt':
        data = getvar(wrf_in, element, units='m2 s-2', timeidx=timeidx)
    if element == 'va' or element == 'ua':
        data = getvar(wrf_in, element, units='m s-1', timeidx=timeidx)
    if element == 'temp':
        data = getvar(wrf_in, element, units='K', timeidx=timeidx)
    # else:
    #     data = getvar(wrf_in, element, units=units, timeidx=timeidx)

    data_pressure_levels = interplevel(data, pressure, pressure_levels)
    # 处理
    return data_pressure_levels


def WRF_data_process(WRF_file, element, units=None, timeidx=0):

    wrf = Dataset(WRF_file)
    if element == "longitude":
        data = to_np(wrf.variables['XLONG'][0, :, :])
    elif element == 'latitude':
        data = to_np(wrf.variables['XLAT'][0, :, :])
    elif element == 'temp':
        data = getvar(wrf, element, units=units,
                      timeidx=timeidx)
    elif element == 'rain':
        rainnc = getvar(wrf, "RAINNC", timeidx=timeidx)
        rainc = getvar(wrf, "RAINC", timeidx=timeidx)
        data = rainnc + rainc
    else:
        data = getvar(wrf, element, units=units,timeidx=timeidx)

    dataold = to_np(data)
    return dataold


def EC_data_process(EC_file, element, units=None, ec_start_time=0, ec_end_time=0, time_interval=1):
    """

    :param EC_file:
    :param ec_start_time:
    :param ec_end_time:
    :param time_interval:
    :param element:
    :return:
    """
    EC_in = Dataset(EC_file)
    if element == "longitude":
        data = to_np(EC_in.variables[element])[:]-180
    else:
        data = to_np(EC_in.variables[element])
    # levels = to_np(EC_in.variables["levels"])
    return data


def interp_EC_pressure_level_in_wrf_gird(EC_dir, WRF_dir, EC_element, timeidx, save_dir=None):
    """

    :param EC_dir:
    :param EC_element:
    :param timeidx:
    :param save_dir:
    :return:
    """
    t = EC_data_process(EC_dir, EC_element)  # [:,:,:,:]
    time = EC_data_process(EC_dir, "time")
    EC_UTC = utils.ECMWF_to_UTC(time)
    forecast_time = timeidx
    EC_start_idx = EC_UTC.index(forecast_time.strftime('%Y-%m-%d_%H_%M_%S'))

    xec = EC_data_process(EC_f_dir, "longitude")
    yec = EC_data_process(EC_f_dir, "latitude")
    xec, yec = np.meshgrid(xec, yec)


    xwrf = WRF_data_process(WRF_dir, "longitude")
    ywrf = WRF_data_process(WRF_dir, "latitude")

    # t_flatten = np.ma.resize(t, (t.shape[0]*t.shape[1], t.shape[2], t.shape[3]))
    t_flatten = t[EC_start_idx, :, :, :]
    t_interp_flatten = utils.interpolate_3d(t_flatten, yec, xec, ywrf, xwrf)
    out_dict = {"result": t_interp_flatten}
    if save_dir != None:
        with open(save_dir, 'wb') as o:
            pickle.dump(out_dict, o)
        return 0
    else:
        return t_interp_flatten


def RMSE_on_Z_level(pre, obs):
    RMSE = np.sqrt((obs - pre) ** 2)
    return np.mean(RMSE, axis=(1, 2))


def test(WRF_dir, EC_dir, wrf_element, EC_element, time, save_dir=None):
    t = EC_data_process(EC_dir, "t")# [:,:,:,:]
    time = EC_data_process(EC_dir, "time")
    EC_UTC = utils.ECMWF_to_UTC(time)
    forecast_time = dt.datetime(2021, 8, 30, 0, 0, 0)
    EC_start_idx = EC_UTC.index(forecast_time.strftime('%Y-%m-%d_%H_%M_%S'))

    xec = EC_data_process(EC_f_dir, "longitude")
    yec = EC_data_process(EC_f_dir, "latitude")
    xec, yec = np.meshgrid(xec, yec)

    twrf = WRF_data_process(WRF_dir, "temp", units="degC")
    xwrf = WRF_data_process(WRF_dir, "longitude")
    ywrf = WRF_data_process(WRF_dir, "latitude")

    # t_flatten = np.ma.resize(t, (t.shape[0]*t.shape[1], t.shape[2], t.shape[3]))
    t_flatten = t[EC_start_idx, :, :, :]
    t_interp_flatten = utils.interpolate_3d(t_flatten, yec, xec, ywrf, xwrf)



    s=1


# test

# EC_elements = ['z', 'u', 'v']
# WRF_elements = ['z', 'ua', 'va']
EC_elements = ['r', 't', 'z', 'u']
WRF_elements = ['rh', 'temp', 'z', 'ua']
solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
ext = [100, 300]
EC_el = 't'
WRF_el = 'temp'
EC_f_dir = r'F:\WRF\DOUBLE_V1\EC\pressure_level\adaptor.mars.internal-1667377694.273551-4252-18-b4b673e7-b9a5-4104-811e-12a18631fd65.nc'
WRF_f_dir = r'F:\WRF\DOUBLE_V1\CON0\wrfout_d01_2021-08-29_00_00_00.nc'
save_dir = r'F:\WRF\DOUBLE_V1\pressure_level_comp'
pressure_levels = EC_data_process(EC_f_dir, 'level')[:]
# wrf_fore_time = [0, 6, 12, 16, 18]
wrf_fore_time = [8, 12]
forecast_time = [dt.datetime(2021, 8, 29, 0, 0, 0)+dt.timedelta(hours=each*3) for each in wrf_fore_time]
# test(WRF_f_dir, EC_f_dir, save_dir)
rmse_sheet = pd.DataFrame(np.zeros(shape=(4, 16)), columns=pressure_levels, index=solution)
# for wrf_f, ec_f in zip(wrf_fore_time, forecast_time):
#     for ec_e, wrf_e in zip(EC_elements, WRF_elements):
#         for i in solution:
#             WRFf = r'F:\WRF\DOUBLE_V1' + '\\' + i + r'\wrfout_d01_2021-08-29_00_00_00.nc'
#             ec_interp = interp_EC_pressure_level_in_wrf_gird(EC_f_dir, WRF_f_dir, ec_e, ec_f)
#             wrf_in_pressure_levels = interp_on_pressure_level(WRFf, wrf_e, None, wrf_f, pressure_levels)
#             rmse_sheet.loc[i, :] = RMSE_on_Z_level(wrf_in_pressure_levels[:,ext[0]:ext[1],ext[0]:ext[1]], ec_interp[:,ext[0]:ext[1],ext[0]:ext[1]])[:]
#         rmse_sheet.to_csv(r'F:\WRF\DOUBLE_V1\pressure_level_comp\\'+ec_e+'_rmse_'+str(wrf_f)+'.csv')

list1 = utils.gen_list_2_var(r'F:\WRF\DOUBLE_V1\pressure_level_comp\{}_rmse_{}.csv',
                             EC_elements,
                             wrf_fore_time)
# list1 = ['t_rmse.csv', 'rh_rmse.csv', 'v_rmse.csv', 'z_rmse.csv',
#          't_rmse_72.csv', 'rh_rmse_72.csv', 'v_rmse_72.csv', 'z_rmse_72.csv']
csvlist = list1

plot.RMSE_sub_plot((len(wrf_fore_time), len(EC_elements)), csvlist, labels=EC_elements*len(wrf_fore_time),
                   savedir=r"F:\WRF\DOUBLE_V1\pressure_level_comp")

s = 1



