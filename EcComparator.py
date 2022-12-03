# -*- encoding: utf-8 -*-
'''
与真实值进行比较
@File    :   EcComparator.py
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/4/1 15:17   WangQiuyi      1.0         None
'''

from scipy.interpolate import interp2d as itp2
import os
import numpy as np
import netCDF4 as nc
from wrf import to_np, getvar
import utils
import wrf_classes
import pickle
import plot
import pandas as pd
import eccodes
import datetime as dt
import xarray as xr


def PostProcessECWRF(EC_config: dict, wrf_config: dict, element, out_path: str):
    """
    wrf和ECMWF数据预处理
    :param out_path:
    :param element:
    :param EC_config: EC配置
    :param wrf_config: WRF配置
    :return:
    """
    if element == "rain":
        wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
                                        step=wrf_config['step'],
                                        interval=wrf_config['outputinterval'])
        if EC_config['file_mode'] == 'single' and wrf_config['file_mode'] == 'single':
            # single file 模式的数据预处理
            ECMWF_file_dir = EC_config['dir']
            wrf_file_dir = wrf_config['dir']
            # ECMWF
            ec = nc.Dataset(ECMWF_file_dir)
            x_ecold = to_np(ec.variables['longitude'])  # 0-360
            x_ec = x_ecold[:] - 180
            y_ecold = to_np(ec.variables['latitude'])
            y_ec = y_ecold[:]
            x_ec, y_ec = np.meshgrid(x_ec, y_ec)

            time = to_np(ec.variables['time'])  #
            EC_UTC = utils.ECMWF_to_UTC(time)
            # 提取合适的EC时间和空间区间
            EC_start_idx = EC_UTC.index(wrf_time.stime.strftime('%Y-%m-%d_%H_%M_%S'))
            EC_end_idx = EC_UTC.index(wrf_time.etime.strftime('%Y-%m-%d_%H_%M_%S'))
            # z_ec 有三个维度 时间， 经度， 纬度
            z_ec_origin = to_np(ec.variables[EC_config['element']])[EC_start_idx:(EC_end_idx+1), :, :]
            z_ec = np.zeros(shape=(wrf_config['step']+1,x_ec.shape[0],x_ec.shape[1]))
            # 对WRF output所需要的间隔时间进行累加
            for i in range(wrf_config['step']):
                # temp = np.zeros(shape=(1, x_ec.shape[0], x_ec.shape[1]))
                for j in range(wrf_config['outputinterval']):
                    z_ec[i+1, :, :] = z_ec[i+1, :, :] + z_ec_origin[i*wrf_config['outputinterval']+j, :, :]


            # ------------ WRF --------------------
            wrf = nc.Dataset(wrf_file_dir)
            # wrfinterval = utils.Extract_WRF_interval(wrf)
            # rate = int(wrf_config['outputinterval']/wrf_config['origininterval'])
            # series = range(0, wrf_config['step']*2+2, rate)

            x_wrf = to_np(wrf.variables['XLONG'][0, :, :])  # -180 ~180
            y_wrf = to_np(wrf.variables['XLAT'][0, :, :])
            wrf_file = nc.Dataset(wrf_config['dir'])
            # data = getvar(wrf_file, element, units=units, timeidx=timeidx)
            if wrf_config['element'] == 'temp':
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])
            elif wrf_config['element'] == 'rain':
                rainnc = getvar(wrf_file, "RAINNC", timeidx=wrf_config['timeidx'])
                rainc = getvar(wrf_file, "RAINC", timeidx=wrf_config['timeidx'])
                data = rainnc + rainc
            else:
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])

            dataold = to_np(data)
            datanew = np.zeros(shape=(wrf_config['step']+1, dataold.shape[1], dataold.shape[2]))

            rate = int(wrf_config['outputinterval']/wrf_config['origininterval'])
            for i in range(wrf_config['step']):
                datanew[i + 1] = dataold[i*rate + rate] - dataold[i*rate]



            out_dict = {"wrf_lon": x_wrf,
                        "wrf_lat": y_wrf,
                        "wrf_data": datanew[:, :, :],
                        "EC_lon": x_ec,
                        "EC_lat": y_ec,
                        "EC_date": z_ec * 1000,
                        "time": wrf_time}
            # np.save(out_path, out_dict)
            with open(out_path, 'wb') as e:
                pickle.dump(out_dict, e)

    if element == "td2":
        K = -273.15
        wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
                                        step=wrf_config['step'],
                                        interval=wrf_config['outputinterval'])
        if EC_config['file_mode'] == 'single' and wrf_config['file_mode'] == 'single':
            # single file 模式的数据预处理
            ECMWF_file_dir = EC_config['dir']
            wrf_file_dir = wrf_config['dir']
            # ECMWF
            ec = nc.Dataset(ECMWF_file_dir)
            x_ecold = to_np(ec.variables['longitude'])  # 0-360
            x_ec = x_ecold[:] - 180
            y_ecold = to_np(ec.variables['latitude'])
            y_ec = y_ecold[:]
            x_ec, y_ec = np.meshgrid(x_ec, y_ec)

            time = to_np(ec.variables['time'])  #
            EC_UTC = utils.ECMWF_to_UTC(time)
            # 提取合适的EC时间和空间区间
            EC_start_idx = EC_UTC.index(wrf_time.stime.strftime('%Y-%m-%d_%H_%M_%S'))
            EC_end_idx = EC_UTC.index(wrf_time.etime.strftime('%Y-%m-%d_%H_%M_%S'))
            # z_ec 有三个维度 时间， 经度， 纬度

            temp = ec.variables[EC_config['element']][2, :, :]
            z_ec_origin = to_np(ec.variables[EC_config['element']])[EC_start_idx:(EC_end_idx + 1), :, :]
            z_ec = np.zeros(shape=(wrf_config['step'] + 1, x_ec.shape[0], x_ec.shape[1]))

            for i in range(wrf_config['step']):
                z_ec[i, :, :] = z_ec_origin[i*wrf_config['outputinterval'], :, :]


            # 对WRF output所需要的间隔时间进行累加
            # ------------ WRF --------------------
            wrf = nc.Dataset(wrf_file_dir)
            # wrfinterval = utils.Extract_WRF_interval(wrf)
            # rate = int(wrf_config['outputinterval']/wrf_config['origininterval'])
            # series = range(0, wrf_config['step']*2+2, rate)

            x_wrf = to_np(wrf.variables['XLONG'][0, :, :])  # -180 ~180
            y_wrf = to_np(wrf.variables['XLAT'][0, :, :])
            wrf_file = nc.Dataset(wrf_config['dir'])

            # data = getvar(wrf_file, element, units=units, timeidx=timeidx)
            if wrf_config['element'] == 'temp':
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])
            elif wrf_config['element'] == 'rain':
                rainnc = getvar(wrf_file, "RAINNC", timeidx=wrf_config['timeidx'])
                rainc = getvar(wrf_file, "RAINC", timeidx=wrf_config['timeidx'])
                data = rainnc + rainc
            else:
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])

            dataold = to_np(data)

            out_dict = {"wrf_lon": x_wrf,
                        "wrf_lat": y_wrf,
                        "wrf_data": dataold[:, :, :],
                        "EC_lon": x_ec,
                        "EC_lat": y_ec,
                        "EC_date": z_ec + K,
                        "time": wrf_time}
            # np.save(out_path, out_dict)
            with open(out_path, 'wb') as e:
                pickle.dump(out_dict, e)

    if element == "u10" or element == "v10":
        K = -273.15
        wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
                                        step=wrf_config['step'],
                                        interval=wrf_config['outputinterval'])
        if EC_config['file_mode'] == 'single' and wrf_config['file_mode'] == 'single':
            # single file 模式的数据预处理
            ECMWF_file_dir = EC_config['dir']
            wrf_file_dir = wrf_config['dir']
            # ECMWF
            ec = nc.Dataset(ECMWF_file_dir)
            x_ecold = to_np(ec.variables['longitude'])  # 0-360
            x_ec = x_ecold[:] - 180
            y_ecold = to_np(ec.variables['latitude'])
            y_ec = y_ecold[:]
            x_ec, y_ec = np.meshgrid(x_ec, y_ec)

            time = to_np(ec.variables['time'])  #
            EC_UTC = utils.ECMWF_to_UTC(time)
            # 提取合适的EC时间和空间区间
            EC_start_idx = EC_UTC.index(wrf_time.stime.strftime('%Y-%m-%d_%H_%M_%S'))
            EC_end_idx = EC_UTC.index(wrf_time.etime.strftime('%Y-%m-%d_%H_%M_%S'))
            # z_ec 有三个维度 时间， 经度， 纬度

            temp = ec.variables[EC_config['element']][2, :, :]
            z_ec_origin = to_np(ec.variables[EC_config['element']])[EC_start_idx:(EC_end_idx + 1), :, :]
            z_ec = np.zeros(shape=(wrf_config['step'] + 1, x_ec.shape[0], x_ec.shape[1]))

            for i in range(wrf_config['step']):
                z_ec[i, :, :] = z_ec_origin[i*wrf_config['outputinterval'], :, :]


            # 对WRF output所需要的间隔时间进行累加
            # ------------ WRF --------------------
            wrf = nc.Dataset(wrf_file_dir)
            # wrfinterval = utils.Extract_WRF_interval(wrf)
            # rate = int(wrf_config['outputinterval']/wrf_config['origininterval'])
            # series = range(0, wrf_config['step']*2+2, rate)

            x_wrf = to_np(wrf.variables['XLONG'][0, :, :])  # -180 ~180
            y_wrf = to_np(wrf.variables['XLAT'][0, :, :])
            wrf_file = nc.Dataset(wrf_config['dir'])
            # data = getvar(wrf_file, element, units=units, timeidx=timeidx)
            if wrf_config['element'] == 'temp':
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])
            elif wrf_config['element'] == 'rain':
                rainnc = getvar(wrf_file, "RAINNC", timeidx=wrf_config['timeidx'])
                rainc = getvar(wrf_file, "RAINC", timeidx=wrf_config['timeidx'])
                data = rainnc + rainc
            else:
                data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'], timeidx=wrf_config['timeidx'])

            dataold = to_np(data)

            out_dict = {"wrf_lon": x_wrf,
                        "wrf_lat": y_wrf,
                        "wrf_data": dataold[:, :, :],
                        "EC_lon": x_ec,
                        "EC_lat": y_ec,
                        "EC_date": z_ec + K,
                        "time": wrf_time}
            # np.save(out_path, out_dict)
            with open(out_path, 'wb') as e:
                pickle.dump(out_dict, e)

        if element == "water vapor":
            K = -273.15
            wrf_time = wrf_classes.WrfTimes(stime=wrf_config['stime'],
                                            step=wrf_config['step'],
                                            interval=wrf_config['outputinterval'])
            if EC_config['file_mode'] == 'single' and wrf_config['file_mode'] == 'single':
                # single file 模式的数据预处理
                ECMWF_file_dir = EC_config['dir']
                wrf_file_dir = wrf_config['dir']
                # ECMWF
                ec = nc.Dataset(ECMWF_file_dir)
                x_ecold = to_np(ec.variables['longitude'])  # 0-360
                x_ec = x_ecold[:] - 180
                y_ecold = to_np(ec.variables['latitude'])
                y_ec = y_ecold[:]
                x_ec, y_ec = np.meshgrid(x_ec, y_ec)

                time = to_np(ec.variables['time'])  #
                EC_UTC = utils.ECMWF_to_UTC(time)
                # 提取合适的EC时间和空间区间
                EC_start_idx = EC_UTC.index(wrf_time.stime.strftime('%Y-%m-%d_%H_%M_%S'))
                EC_end_idx = EC_UTC.index(wrf_time.etime.strftime('%Y-%m-%d_%H_%M_%S'))
                # z_ec 有三个维度 时间， 经度， 纬度

                temp = ec.variables[EC_config['element']][2, :, :]
                z_ec_origin = to_np(ec.variables[EC_config['element']])[EC_start_idx:(EC_end_idx + 1), :, :]
                z_ec = np.zeros(shape=(wrf_config['step'] + 1, x_ec.shape[0], x_ec.shape[1]))

                for i in range(wrf_config['step']):
                    z_ec[i, :, :] = z_ec_origin[i * wrf_config['outputinterval'], :, :]

                # 对WRF output所需要的间隔时间进行累加
                # ------------ WRF --------------------
                wrf = nc.Dataset(wrf_file_dir)
                # wrfinterval = utils.Extract_WRF_interval(wrf)
                # rate = int(wrf_config['outputinterval']/wrf_config['origininterval'])
                # series = range(0, wrf_config['step']*2+2, rate)

                x_wrf = to_np(wrf.variables['XLONG'][0, :, :])  # -180 ~180
                y_wrf = to_np(wrf.variables['XLAT'][0, :, :])
                wrf_file = nc.Dataset(wrf_config['dir'])
                # data = getvar(wrf_file, element, units=units, timeidx=timeidx)
                if wrf_config['element'] == 'temp':
                    data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'],
                                  timeidx=wrf_config['timeidx'])
                elif wrf_config['element'] == 'rain':
                    rainnc = getvar(wrf_file, "RAINNC", timeidx=wrf_config['timeidx'])
                    rainc = getvar(wrf_file, "RAINC", timeidx=wrf_config['timeidx'])
                    data = rainnc + rainc
                else:
                    data = getvar(wrf_file, wrf_config['element'], units=wrf_config['units'],
                                  timeidx=wrf_config['timeidx'])

                dataold = to_np(data)

                out_dict = {"wrf_lon": x_wrf,
                            "wrf_lat": y_wrf,
                            "wrf_data": dataold[:, :, :],
                            "EC_lon": x_ec,
                            "EC_lat": y_ec,
                            "EC_date": z_ec + K,
                            "time": wrf_time}
                # np.save(out_path, out_dict)
                with open(out_path, 'wb') as e:
                    pickle.dump(out_dict, e)

    return 1


def ECMWF_WRF_comparator(ECMWF_config, wrf_config, out_file, kind: str = 'linear'):
    """
    用于EC和WRF之间进行比较
    ECMWF_x, ECMWF_y, wrf_x, wrf_y 只需要是nc文件自带的结果就行
    ！！！interp2d()中，输入的x,y,z需要先用ravel()被转成了一维数组
    :param out_file:
    :param wrf_config:
    :param ECMWF_config, wrf_config EC&WRF配置
    :param kind: 差值方式默认双线性
    :return: 差值结果
    """
    # 1. 数据前处理
    # ----------- ECWMF -------------------
    PostProcessECWRF(ECMWF_config, wrf_config, 'rain', out_file)
    with open(out_file, 'rb') as o:
        try:
            data = pickle.load(o)
        except EOFError:
            return None
    # 差值

    wrf_fore_time_num=25
    """reshape 栅格数据, 先展平后折叠"""
    EC = np.append(data['EC_date'], np.append(data['EC_lon'], data['EC_lat']))
    WRF_COOR = np.append(data['wrf_lon'], data['wrf_lat'])
    EC = EC.reshape((data['EC_date'].shape[0] + 2, data['EC_date'].shape[1], data['EC_date'].shape[2]))
    WRF = WRF_COOR.reshape((2, data['wrf_lon'].shape[0], data['wrf_lon'].shape[1]))

    out = utils.ec_wrf_bilinear_interpolate(EC, WRF, wrf_config)
    with open(os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_in_WRF_grid.pkl'), 'rb') as o:
        out = pickle.load(o)

    # 保存一次变量

    with open(os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_WRF_in_same_grid.pkl'), 'wb') as e:
        out_dict = {"EC":  out['result'][0:wrf_fore_time_num, :, :],
                    "WRF": data['wrf_data'],
                    "lon": data['wrf_lon'],
                    "lat": data['wrf_lat']}
        try:
            pickle.dump(out_dict, e)
        except EOFError:
            None

    # WRF 中的数据是三个小时的累积数据， EC是每个小时的降水数据

    pd.DataFrame(out['result'][6, :, :]).to_csv(os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_in_WRF_grid.csv'))
    pd.DataFrame(data['wrf_data'][6, :, :]-out['result'][6, :, :]).to_csv(os.path.join(os.path.split(wrf_config['dir'])[0], 'wrf_data_EC.csv'))
    delta = data['wrf_data'] - out['result'][0:wrf_fore_time_num, :, :]

    delta_path = os.path.join(os.path.split(wrf_config['dir'])[0], 'Delta_between_WRF_and_EC.pkl')
    with open(delta_path, 'wb') as e:
        try:
            pickle.dump(delta_path, e)
        except EOFError:
            None

    # 计算 RMSE


    for i in range(wrf_fore_time_num):
        dirname = "td2&EC"
        plot.element_plot_v2(wrf_config['dir'], delta[0:wrf_fore_time_num, :, :], dirname,
                             [-105, -75, 10, 40],
                             timeidx=i,
                             levels=np.arange(-40, 40, 1),
                             yticks=[10, 20, 30, 40],
                             xticks=[-105, -95, -85, -75],
                             # mask=[-50, 50],
                             )
    # for i in range(data['time'].step+1):
    #     func = itp2(x=data['EC_lon'], y=data['EC_lat'], z=data['EC_date'][i, :, :], kind=kind)
    #     z_out[i, :, :] = func(data['wrf_lon'], data['wrf_lat'])
    #     s = 1
    s = 1

    # 输出


def ECMWF_GPSPWV_comparator(fera5, site_grid):
    """
    对比ECMWF在分析资料中的PWV和GNSS点的PWV值
    :return:
    """
    lon_grid = utils.faker_lon(site_grid["LON"])
    lat_grid = site_grid["LAT"]
    hgt_grid = site_grid["HEIGHT"]
    name_grid = site_grid["NAME"]

    file_suffix = fera5.split(".")[-1]
    if file_suffix == 'grib':
        ds = xr.open_dataset(fera5, engine='cfgrib')
        time = ds.variables['time']
        pwv = ds.tcwv
        z = ds.z
        # lon_ecold = to_np(ds.variables['longitude'])  # 0-360
        # lon = lon_ecold[:]
        # lat_ecold = to_np(ds.variables['latitude'])
        # lat = lat_ecold[:]
        # x, y = np.meshgrid(lon, lat)
        tgt_lon = xr.DataArray(lon_grid.values, dims='tcwv')
        tgt_lat = xr.DataArray(lat_grid.values, dims='tcwv')
        site_pwv = pwv.sel(longitude=tgt_lon, latitude=tgt_lat)

        a = 1

        # site_ec_pwv, site_ec_z = [], []
        # # 对每个目标点进行循环
        # for i in range(len(lon_grid)):
        #     # 0 目标点的BLH
        #     lon, lat, hgt = lon_grid[i], lat_grid[i], hgt_grid[i]
        #     # 1 读取最临近的四个点的BLH，高程，PWV值
        #     ec_z = utils.get_grib_point_coordinates(eccodes.codes_grib_find_nearest(gidz,   lat, lon, False, 4))
        #     ec_pwv = utils.get_grib_point_coordinates(eccodes.codes_grib_find_nearest(gidpwv, lat, lon, False, 4))
        #     # 2 双线性内插
        #     site_ec_pwv.append(utils.bilinear_interpolation_for_1_point(ec_pwv[:, :], [lon, lat]))
        #     site_ec_z.append(utils.bilinear_interpolation_for_1_point(ec_z[:, :], [lon, lat]))

    if file_suffix == 'nc':
        stime = dt.datetime(2021, 8, 29, 00, 00, 00)
        ec = nc.Dataset(fera5)
        x_ecold = to_np(ec.variables['longitude'])  # 0-360
        x_ec = x_ecold[:] - 180
        y_ecold = to_np(ec.variables['latitude'])
        y_ec = y_ecold[:]
        x_ec, y_ec = np.meshgrid(x_ec, y_ec)

        time = to_np(ec.variables['time'])  #
        EC_UTC = utils.ECMWF_to_UTC(time)
        s = 1
        # 提取合适的EC时间和空间区间
        # EC_start_idx = EC_UTC.index(wrf_time.stime.strftime('%Y-%m-%d_%H_%M_%S'))
        # EC_end_idx = EC_UTC.index(wrf_time.etime.strftime('%Y-%m-%d_%H_%M_%S'))
    s = 1


def Pwv_Height_correction(pwv_value, origin_height, target_height):
    """
    对PWV值进行高程上的改正
    :return:
    """
    dif_hgt = origin_height / 9.80665 - target_height
    era5_wv_cor = pwv_value * np.exp(0.439 * dif_hgt / 1000)
    return era5_wv_cor
