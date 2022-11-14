# -*- encoding: utf-8 -*-
'''
@File    :   utils.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/4/1 17:24   WangQiuyi      1.0         None
'''

import datetime as dt
import numpy as np
import pickle
import os


def Extract_WRF_interval(wrffile):
    tc0 = wrffile.variables['Times'][0]
    tc1 = wrffile.variables['Times'][1]
    t0, t1 = b'', b''
    for i in tc0:
        t0 = t0 + i
    for j in tc1:
        t1 = t1 + j
    t0 = dt.datetime.strptime(t0.decode(), "%Y-%m-%d_%H:%M:%S")
    t1 = dt.datetime.strptime(t1.decode(), "%Y-%m-%d_%H:%M:%S")
    dt0 = t1 - t0
    return int(dt0.seconds / 3600)


def ECMWF_to_UTC(EC_time):
    T0 = dt.datetime(1900, 1, 1, 0, 0)
    # ECt0 = T0 + dt.timedelta(hours=int(EC_time[0]))
    UTCtime = []
    # for i in range(len(EC_time)):
    #     UTCtime.append(ECt0.strftime("%Y-%m-%d_%H_%M_%S"))
    #     ECt0 = ECt0 + dt.timedelta(hours=1)
    for each in EC_time:
        ECt = T0 + dt.timedelta(hours=int(each))
        UTCtime.append(ECt.strftime("%Y-%m-%d_%H_%M_%S"))

    return UTCtime


def flood_index(x, x_range):
    interval = x_range[1] - x_range[0]
    if interval < 0:
        x_max = np.max(x_range)
        out = np.array([np.floor((each - x_max) / interval) for each in x], dtype=np.int)
    else:
        x_min = np.min(x_range)
        out = np.array([np.floor((each - x_min) / interval) for each in x], dtype=np.int)
    return out


def bilinear_interpolate(src, dst, wrf_config):
    """
    双线性差值
    :param src:
    :param dst:
    :return:
    """

    level_num = src.shape[0]
    lon_dst, lat_dst = np.ravel(dst[0, :, :]), np.ravel(dst[1, :, :])  # (h, w)
    EC_lon_series, EC_lat_series = src[-2, 0, :], src[-1, :, 0]

    """计算临近点在EC中的坐标"""
    lon_dst_dix = flood_index(lon_dst, EC_lon_series)
    lat_dst_dix = flood_index(lat_dst, EC_lat_series)
    interval = 0.25 * 0.25

    """四个临近点的像素值"""
    f00 = src[:, lat_dst_dix, lon_dst_dix]  # d00
    f01 = src[:, lat_dst_dix + 1, lon_dst_dix]  # d01
    f10 = src[:, lat_dst_dix, lon_dst_dix + 1]  # d10
    f11 = src[:, lat_dst_dix + 1, lon_dst_dix + 1]  # d11

    w00 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d00
    w01 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix + 1])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d01
    w10 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix + 1]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d10
    w11 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix + 1]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix + 1])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d11
    out = f00 * w00.T + f01 * w01.T + f10 * w10.T + f11 * w11.T
    out_dict = {"result": out.reshape((level_num, dst.shape[1], dst.shape[2]))}
    outdir = os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_in_WRF_grid.pkl')
    with open(outdir, 'wb') as o:
        pickle.dump(out_dict, o)
    return out.reshape((level_num, dst.shape[1], dst.shape[2]))


def interpolate_3d(src, src_lat, src_lon, dst_lat, dst_lon, save_dir=None):
    """
    双线性差值
    :param src:
    :param dst:
    :return:
    """

    level_num = src.shape[0]
    lon_dst, lat_dst = np.ravel(dst_lon[:, :]), np.ravel(dst_lat[:, :])  # (h, w)
    EC_lon_series, EC_lat_series = src_lon[0, :], src_lat[:, 0]

    """计算临近点在EC中的坐标"""
    lon_dst_dix = flood_index(lon_dst, EC_lon_series)
    lat_dst_dix = flood_index(lat_dst, EC_lat_series)
    interval = 0.25 * 0.25

    """四个临近点的像素值"""
    f00 = src[:, lat_dst_dix, lon_dst_dix]  # d00
    f01 = src[:, lat_dst_dix + 1, lon_dst_dix]  # d01
    f10 = src[:, lat_dst_dix, lon_dst_dix + 1]  # d10
    f11 = src[:, lat_dst_dix + 1, lon_dst_dix + 1]  # d11

    w00 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d00
    w01 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix + 1])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d01
    w10 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix + 1]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d10
    w11 = np.repeat(np.abs((lon_dst - EC_lon_series[lon_dst_dix + 1]) * \
                           (lat_dst - EC_lat_series[lat_dst_dix + 1])) / interval, level_num, axis=0).reshape(
        (lon_dst.shape[0], level_num))  # d11
    out = f00 * w00.T + f01 * w01.T + f10 * w10.T + f11 * w11.T
    out_dict = {"result": out.reshape((level_num, dst_lon.shape[0], dst_lon.shape[1]))}
    # outdir = os.path.join(os.path.split(wrf_config['dir'])[0], 'EC_in_WRF_grid.pkl')
    if save_dir != None:
        with open(save_dir, 'wb') as o:
            pickle.dump(out_dict, o)
        return 0
    else:
        return out.reshape((level_num, dst_lon.shape[0], dst_lon.shape[1]))


# def bilinear_interpolate(src, dst_size):
#     height_src, width_src, channel_src = src.shape  # (h, w, ch)
#     height_dst, width_dst = np.ravel(dst_size[0, :, :]), np.ravel(dst_size[1, :, :])  # (h, w)
#
#
#
#     """找出每个投影点在原图的近邻点坐标"""
#     ws_0 = np.clip(np.floor(ws_p), 0, width_src - 2).astype(np.int)
#     hs_0 = np.clip(np.floor(hs_p), 0, height_src - 2).astype(np.int)
#     ws_1 = ws_0 + 1
#     hs_1 = hs_0 + 1
#
#     """四个临近点的像素值"""
#     f_00 = src[hs_0, ws_0, :].T
#     f_01 = src[hs_0, ws_1, :].T
#     f_10 = src[hs_1, ws_0, :].T
#     f_11 = src[hs_1, ws_1, :].T
#
#     """计算权重"""
#     w_00 = ((hs_1 - hs_p) * (ws_1 - ws_p)).T
#     w_01 = ((hs_1 - hs_p) * (ws_p - ws_0)).T
#     w_10 = ((hs_p - hs_0) * (ws_1 - ws_p)).T
#     w_11 = ((hs_p - hs_0) * (ws_p - ws_0)).T
#
#     """计算目标像素值"""
#     return (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T


def gen_list_2_var(pattern, v1, v2):
    l = []
    for j in v2:
        for i in v1:
            l.append(pattern.format(i, j))
    return l
