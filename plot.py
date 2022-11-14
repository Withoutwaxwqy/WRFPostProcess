# -*- encoding: utf-8 -*-
'''
@File    :   plot.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/4/13 15:27   WangQiuyi      1.0         None
'''

import numpy as np
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
from wrf import getvar, to_np, latlon_coords, vertcross, smooth2d
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import copy
import cartopy.crs as crs
import cartopy.feature as cfeature
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter


import matplotlib as mpl
mpl.rcParams["font.family"] = 'Arial'  # 默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm'  # 数学文字字体
mpl.rcParams["font.size"] = 12  # 字体大小
mpl.rcParams["axes.linewidth"] = 1  # 轴线边框粗细（默认的太粗了）


def element_plot_v2(wrf_path, comparator, element, extent, timeidx, levels=None, xticks=None, yticks=None, mask=None, bounds_colors=None):
    """

    :param wrf_path: WRF文件地址，用于确定经纬度点的信息
    :param comparator: 绘图数据
    :param element: 绘图元素
    :param extent: 绘图地理范围
    :param timeidx: 时间序列号
    :param levels: 绘图levels
    :param xticks: 绘图x轴标签
    :param yticks: 绘图y轴标签
    :param mask: 数据通道（暂时不用，废弃中）
    :param bounds_colors: 自定义数据颜色对应映射
    :return:
    """
    if mask is None:
        mask = [-np.inf, np.inf]
    if xticks is None:
        xticks = [-90, -88, -86, -84, -82]
    if yticks is None:
        yticks = [26, 28, 30, 32, 34]
    wrf_file = Dataset(wrf_path)
    data = getvar(wrf_file, 'p', units="hPa", timeidx=timeidx)

    comparatordata = comparator[timeidx, :, :]
    proj = crs.PlateCarree()  # 圆柱投影， 默认WGS1984
    # extent = extent

    lats, lons = latlon_coords(data)

    # mask
    upper, lowwer = mask[1], mask[0]
    comparatordata = np.where(comparatordata <= upper-1, comparatordata, upper-1)
    comparatordata = np.where(comparatordata >= lowwer, comparatordata, lowwer)

    fig = plt.figure(figsize=(10, 7.5))
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)
    ax1.set_extent(extent, proj)
    states = NaturalEarthFeature(category='cultural',
                                 scale='50m',
                                 facecolor='none',
                                 name='admin_1_states_provinces_lines')
    ax1.add_feature(states, linewidth=.5)
    ax1.coastlines('50m', linewidth=1)
    # SLP pyplot
    levels = levels
    # c0 = ax1.contour(to_np(lons), to_np(lats), to_np(data),
    #                  levels=levels, colors="black",
    #                  transform=crs.PlateCarree())

    # 若要修改cmap的底层值或者其他值，必须进行深度复制

    # 自定义颜色和对应边界
    # if bounds_colors is None:
    #     bounds = [-50, -25, -10, -2, 2, 10,
    #               20, 30, 40, 45, 50]
    #     colors = ['#015796', '#03a9f4', '#b3e5fc', '#ffffff', '#ffebee',
    #               '#ef9a9a', '#f44336', '#d32f2f', '#8e24aa', '#4a148c']
    #     cmap1 = ListedColormap(colors)
    #     norms1 = BoundaryNorm(bounds, cmap1.N)
    #
    # # cmap1 = copy.copy(get_cmap("bwr"))
    #     c2 = ax1.contourf(to_np(lons), to_np(lats), to_np(comparatordata),
    #                       levels=bounds,
    #                       transform=crs.PlateCarree(),
    #                       cmap=cmap1,
    #                       norm=norms1,
    #                       )
    #     # wind pyplot
    #     c2.cmap.set_over('#4a148c')
    #     c2.cmap.set_under('#6d4c41')

    #     cbar = fig.colorbar(c2, ax=ax1, shrink=.86, spacing='proportional', ticks=bounds)

    c2 = ax1.contourf(to_np(lons), to_np(lats), to_np(comparatordata),
                      transform=crs.PlateCarree(),
                      levels=levels
                      )

    cbar = fig.colorbar(c2, ax=ax1, shrink=.86, spacing='proportional')
    fig_name = os.path.split(wrf_path)[1]
    cbar.set_label(element + " in " + fig_name.split('.')[0][11:30])
    fig_save_path = os.path.join(os.path.split(wrf_path)[0], element, element+'_'+fig_name.split('.')[0] +'_timeidx_'+"{:0>3d}".format(timeidx) +".jpg")
    ax1.set_xticks(xticks, crs=crs.PlateCarree())
    ax1.set_yticks(yticks, crs=crs.PlateCarree())
    # plt.show()
    if not os.path.exists(os.path.split(fig_save_path)[0]):
        os.makedirs(os.path.split(fig_save_path)[0])
    plt.savefig(fig_save_path, dpi=400, pad_inches=0)
    plt.close()


def prep_score_sub_plot(TS_csv_dir, POD_csv_dir, FB_csv_dir, MAR_csv_dir,
                        TS_limit, POD_limit, FB_limit, MAR_limit, mode):
    '''
    
    :param TS_csv_dir: 
    :param POD_csv_dir:
    :param FB_csv_dir: 
    :param MAR_csv_dir: 
    :param TS_limit: 
    :param POD_limit: 
    :param FB_limit: 
    :param MAR_limit: 
    :param mode: fore_time or threshold
    :return: none
    '''

    TS  = pd.read_csv(TS_csv_dir, index_col=0)
    POD = pd.read_csv(POD_csv_dir, index_col=0)
    FB  = pd.read_csv(FB_csv_dir, index_col=0)
    MAR = pd.read_csv(MAR_csv_dir, index_col=0)
    line_style = ["y-", "g-", "b-", "r-"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    lim = [TS_limit, POD_limit, FB_limit, MAR_limit]
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.35)
    score_name = ["TS", "POD", "FSC", "MAR"]
    for i, score in enumerate([TS, POD, FB, MAR]):

        for j in range(4):
            ax[int(i / 2)][i % 2].plot(score.columns, score.loc[solution[j]][:],
                                       line_style[j], label=score.index[j],
                                       linewidth=0.5)
        ax[int(i / 2)][i % 2].set_xlabel(mode)
        # ax[int(i / 2)][i % 2].set_xticks(score.columns[0:2:-1])
        # ax[int(i / 2)][i % 2].set_xticklabels(score.columns[0:2:-1])
        ax[int(i / 2)][i % 2].set_title(score_name[i])
        ax[int(i / 2)][i % 2].legend(loc='best',
                                     ncol=2)

        ax[0][0].text(0.05, 0.1, s="(a)", fontsize=12, transform=ax[0][0].transAxes)
        ax[0][1].text(0.05, 0.1, s="(b)", fontsize=12, transform=ax[0][1].transAxes)
        ax[1][0].text(0.05, 0.1, s="(c)", fontsize=12, transform=ax[1][0].transAxes)
        ax[1][1].text(0.05, 0.85, s="(d)", fontsize=12, transform=ax[1][1].transAxes)


    # plt.show()
    plt.savefig(r'F:\WRF\DOUBLE_V1\sub_plot.png', dpi=600)
    s = 1


def CsvPlot(csv_dir, skill, xticks, savedir):
    df = pd.read_csv(csv_dir, index_col=0)
    PrepSkillDataFramePlot(df, skill, xticks, savedir)


def PrepSkillDataFramePlot(df: pd.DataFrame, skill, savedir=None, xlim=None):
    '''
    根据降雨评分dataframe绘图
    :param df:
    :param skill:
    :param xticks:
    :param savedir:
    :param xlim:
    :return:
    '''
    solutions = df.index
    x = df.columns
    line_style = ["y-", "g-", "b-", "r-"]
    ax = plt.figure(figsize=(5, 4))
    for i in range(len(solutions)):
        plt.plot(x, df.loc[solutions[i]][:],
                 line_style[i], label=solutions[i],
                 linewidth=0.5)
    plt.title(skill)
    plt.xlabel('threshold')
    plt.legend()
    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, skill+".png"), dpi=600)
        else:
            try:
                os.makedirs(savedir)
            except OSError:
                print("Could not create directory")


    s = 1


def RMSE_sub_plot(shape, csv_dir_list, labels, savedir=None):
    line_style = ["y-", "g-", "b-", "r-"]
    solution = ['CON0', 'VAR1', 'VAR2', 'VAR3']
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(15, 15))
    for i in range(shape[0]*shape[1]):
        data = pd.read_csv(csv_dir_list[i], index_col=0)
        for j in range(4):
            ax[int(i / shape[1])][i % shape[1]].plot(data.loc[solution[j]][::-1], data.columns[::-1],
                                                     line_style[j], label=data.index[j],
                                                     linewidth=0.5)
            ax[int(i / shape[1])][i % shape[1]].set_xlabel(labels[i])
            if i % shape[1] == 0:
                ax[int(i / shape[1])][i % shape[1]].set_ylabel("Pressure(hPa)")


            ax[int(i / shape[1])][i % shape[1]].invert_yaxis()
    # ax[0][1].legend(bbox_to_anchor=(1.05, 0), loc='best')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(labels=labels, loc='lower center', ncol=4)
    # plt.title(skill)

    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "sub_rmse.png"), dpi=600)
        else:
            try:
                os.makedirs(savedir)
            except OSError:
                print("Could not create directory")


def vertical_cross_temp_plot(wrf_path, timeidx, start_point, end_point, savedir=None):
    """

    :param wrf_path:
    :param extent:
    :param wdsp_levels:
    :param timeidx:
    :param start_point:
    :param end_point:
    :return:
    """
    ncfile = Dataset(wrf_path)
    # pressure_levels = [300,400,500,700,850,900,925,950,975,1000]
    pressure_levels = np.arange(0.0, 15.0, 0.5)
    # Get the WRF variables
    z = getvar(ncfile, "z", units='km', timeidx=timeidx)
    temp = getvar(ncfile, "temp", units='degC', timeidx=timeidx)
    z_cross = vertcross(temp, z,#  levels=pressure_levels,
                        wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True)
    lats, lons = latlon_coords(temp)
    fig = plt.figure(figsize=(8, 6))
    c= plt.contourf(to_np(z_cross), levels=np.arange(-30, 40, 1),
                 cmap=get_cmap('jet'))
    coord_pairs = to_np(z_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
    plt.xticks(x_ticks[::20], x_labels[::20], rotation=30, fontsize=7)
    plt.yticks(range(0, len(pressure_levels)), pressure_levels)
    plt.colorbar(c)
    plt.ylabel('Pressure (hPa)')
    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "vert_temp_"+str(timeidx)+".png"), dpi=600)
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "vert_temp_" + str(timeidx) + ".png"), dpi=600)
            except OSError:
                print("Could not create directory")


def vertical_cross_wspd_plot(wrf_path, timeidx, start_point, end_point, savedir=None):
    ncfile = Dataset(wrf_path)
    # pressure_levels = [300, 400, 500, 700, 850, 900, 925, 950, 975, 1000]
    pressure_levels = np.arange(0.0, 15.0, 0.5)
    # Get the WRF variables
    z = getvar(ncfile, "z", units='km', timeidx=timeidx)
    wspd = getvar(ncfile, "wspd_wdir", units="kt", timeidx=timeidx)[0, :]
    z_cross = vertcross(wspd, z, # levels=pressure_levels[::-1],
                        wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True)
    lats, lons = latlon_coords(wspd)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    z_cross = z_cross[0:100, :]
    c = plt.contourf(to_np(z_cross), levels=np.arange(0, 120, 5),
                     cmap=get_cmap('rainbow'))
    coord_pairs = to_np(z_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
    plt.xticks(x_ticks[::20], x_labels[::20], rotation=30, fontsize=7)
    wspd_vals = to_np(z_cross.coords["vertical"])
    wspd_ticks = np.arange(wspd_vals.shape[0])
    ax.set_yticks(wspd_ticks[::20])
    ax.set_yticklabels(wspd_vals[::20], fontsize=7)
    cb_wspd = fig.colorbar(c, ax=ax)
    cb_wspd.ax.tick_params(labelsize=5)
    plt.ylabel('height (km)')
    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "vert_wspd_" + str(timeidx) + ".png"), dpi=600)
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "vert_wspd_" + str(timeidx) + ".png"), dpi=600)
            except OSError:
                print("Could not create directory")


def vertical_cross_wspd_subplot(wrf_paths, timeidx, start_point, end_point, savedir=None):
    letterstr = 'abcdefghijklmn'
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    for i, wrf_path in zip(range(len(wrf_paths)), wrf_paths):
        ncfile = Dataset(wrf_path)
        # pressure_levels = [300, 400, 500, 700, 850, 900, 925, 950, 975, 1000]
        pressure_levels = np.arange(0.0, 15.0, 0.5)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        wspd = getvar(ncfile, "wspd_wdir", units="kt", timeidx=timeidx)[0, :]
        z_cross = vertcross(wspd, z,  # levels=pressure_levels[::-1],
                            wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        lats, lons = latlon_coords(wspd)

        z_cross = z_cross[0:100, :]
        c = ax[i].contourf(to_np(z_cross), levels=np.arange(0, 120, 5),
                         cmap=get_cmap('rainbow'))
        coord_pairs = to_np(z_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str(fmt="({:.1f}N, {:.1f}W)") for pair in np.abs(to_np(coord_pairs))]
        ax[i].set_xticks(x_ticks[::20])
        ax[i].set_xticklabels(x_labels[::20], fontsize=12, )
        ax[i].text(0.05, 0.1, s=letterstr[i], fontsize=12, transform=ax[i].transAxes, zorder=3)
    wspd_vals = to_np(z_cross.coords["vertical"])
    wspd_ticks = np.arange(wspd_vals.shape[0])
    ax[0].set_yticks(wspd_ticks[::20])
    ax[0].set_yticklabels(["{:.2f}".format(each) for each in wspd_vals[::20]], fontsize=12)
    cb_wspd = fig.colorbar(c, ax=[ax[i] for i in range(len(wrf_paths))], fraction=0.02, pad=0.05)
    cb_wspd.ax.tick_params(labelsize=12)
    ax[0].set_ylabel('height (km)')
    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "vert_wspd_all_resolution_" + str(timeidx) + ".png"), dpi=600, bbox_inches='tight')
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "vert_wspd_all_resolution_" + str(timeidx) + ".png"), dpi=600, bbox_inches='tight')
            except OSError:
                print("Could not create directory")
    pass


def ctt_subplot(wrf_paths, extent, timeidx, start_point, end_point, savedir=None):
    letterstr = 'abcdefghijklmn'
    proj = crs.PlateCarree()
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          scale='50m',
                                          facecolor='none',
                                          name='admin_1_states_provinces_lines')
    land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                        scale='50m',
                                        facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                         scale='50m',
                                         facecolor=cfeature.COLORS['water'])

    fig = plt.figure(figsize=(30, 6))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, wrf_path in zip(range(len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(1, 4, i+1, projection=proj)
        ax.set_extent(extent, proj)
        ncfile = Dataset(wrf_path)
        # pressure_levels = [300, 400, 500, 700, 850, 900, 925, 950, 975, 1000]
        pressure_levels = np.arange(0.0, 15.0, 0.5)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        ctt = getvar(ncfile, "ctt", timeidx=timeidx)
        slp = getvar(ncfile, "slp", timeidx=timeidx)
        smooth_slp = smooth2d(slp, 3)
        lats, lons = latlon_coords(slp)
        contour_levels = [960, 965, 970, 975, 980, 990]
        c1 = ax.contour(lons, lats, to_np(smooth_slp), levels=contour_levels,
                            colors="white", transform=crs.PlateCarree(), zorder=3,
                            linewidths=1.0)
        contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
        ctt_contours = ax.contourf(to_np(lons), to_np(lats), to_np(ctt),
                                       contour_levels, cmap=get_cmap("Greys"),
                                       transform=crs.PlateCarree(), zorder=2)
        ax.plot([start_point.lon, end_point.lon],
                    [start_point.lat, end_point.lat], color="yellow", marker="o",
                    transform=crs.PlateCarree(), zorder=3)
        ax.add_feature(land)
        ax.add_feature(states, linewidth=.5, edgecolor="black")
        ax.coastlines('50m', linewidth=1)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks(np.arange(extent[1]+2, extent[0], 4), crs=proj)
        if i == 0:
            ax.set_yticks(np.arange(extent[2]+2, extent[3], 4), crs=proj)
        else:
            ax.set_yticks([])
        ax.tick_params(colors='k')

    cb_ax = fig.add_axes([0.9, 0.1, 0.008, 0.8])
    cb_ctt = fig.colorbar(ctt_contours, cax=cb_ax)
    cb_ctt.ax.tick_params(labelsize=12)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
    # fig.tight_layout()
    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "cct_all_solution_" + str(timeidx) + ".png"), dpi=600,
                        bbox_inches='tight')
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "vert_wspd_all_resolution_" + str(timeidx) + ".png"), dpi=600,
                            bbox_inches='tight')
            except OSError:
                print("Could not create directory")
    pass


def ctt_vertwspd_subplot(wrf_paths, extent, timeidx, start_point, end_point, savedir=None):
    letterstr = 'abcdefghijklmn'
    proj = crs.PlateCarree()
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          scale='50m',
                                          facecolor='none',
                                          name='admin_1_states_provinces_lines')
    land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                        scale='50m',
                                        facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                         scale='50m',
                                         facecolor=cfeature.COLORS['water'])

    fig = plt.figure(figsize=(24, 12))
    # fig.subplots_adjust(hspace=0.1, wspace=0.05)
    for i, wrf_path in zip(range(0, len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(2, 4, i + 1, projection=proj)
        ax.set_extent(extent, proj)
        ncfile = Dataset(wrf_path)
        # pressure_levels = [300, 400, 500, 700, 850, 900, 925, 950, 975, 1000]
        pressure_levels = np.arange(0.0, 15.0, 0.5)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        ctt = getvar(ncfile, "ctt", timeidx=timeidx)
        slp = getvar(ncfile, "slp", timeidx=timeidx)
        smooth_slp = smooth2d(slp, 3)
        lats, lons = latlon_coords(slp)
        contour_levels = [960, 965, 970, 975, 980, 990]
        c1 = ax.contour(lons, lats, to_np(smooth_slp), levels=contour_levels,
                        colors="white", transform=crs.PlateCarree(), zorder=3,
                        linewidths=1.0)
        contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
        ctt_contours = ax.contourf(to_np(lons), to_np(lats), to_np(ctt),
                                   contour_levels, cmap=get_cmap("Greys"),
                                   transform=crs.PlateCarree(), zorder=2)
        ax.plot([start_point.lon, end_point.lon],
                [start_point.lat, end_point.lat], color="yellow", marker="o",
                transform=crs.PlateCarree(), zorder=3)
        ax.add_feature(land)
        ax.add_feature(states, linewidth=.5, edgecolor="black")
        ax.coastlines('50m', linewidth=1)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks(np.arange(extent[1] + 2, extent[0], 4), crs=proj)
        if i == 0:
            ax.set_yticks(np.arange(extent[2] + 2, extent[3], 4), crs=proj)
        else:
            ax.set_yticks([])
        ax.tick_params(colors='k')

    cb_ax = fig.add_axes([0.92, 0.56, 0.008, 0.3])
    cb_ctt = fig.colorbar(ctt_contours, cax=cb_ax)
    cb_ctt.ax.tick_params(labelsize=12)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
    # fig.tight_layout()

    for i, wrf_path in zip(range(0, len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(2, 4, i + 1+len(wrf_paths))

        ncfile = Dataset(wrf_path)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        wspd = getvar(ncfile, "wspd_wdir", units="kt", timeidx=timeidx)[0, :]
        z_cross = vertcross(wspd, z,  # levels=pressure_levels[::-1],
                            wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        lats, lons = latlon_coords(wspd)

        z_cross = z_cross[0:100, :]
        c = ax.contourf(to_np(z_cross), levels=np.arange(0, 120, 5),
                           cmap=get_cmap('rainbow'))
        coord_pairs = to_np(z_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str(fmt="({:.1f}N, {:.1f}W)") for pair in np.abs(to_np(coord_pairs))]
        ax.set_xticks(x_ticks[::20])
        ax.set_xticklabels(x_labels[::20], fontsize=10)
        # ax.text(0.05, 0.1, s=letterstr[i], fontsize=12, transform=ax.transAxes, zorder=3)
        if i == 0:
            wspd_vals = to_np(z_cross.coords["vertical"])
            wspd_ticks = np.arange(wspd_vals.shape[0])
            ax.set_yticks(wspd_ticks[::20])
            ax.set_yticklabels(["{:.2f}".format(each) for each in wspd_vals[::20]], fontsize=12)
    vt_ax = fig.add_axes([0.92, 0.13, 0.008, 0.3])
    cb_vt = fig.colorbar(c, cax=vt_ax)


    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "cct_verttemp_all_solution_" + str(timeidx) + ".png"), dpi=600,
                        bbox_inches='tight')
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "cct_verttemp_all_resolution_" + str(timeidx) + ".png"), dpi=600,
                            bbox_inches='tight')
            except OSError:
                print("Could not create directory")
    pass


def ctt_vert_wspd_temp_subplot(wrf_paths, extent, timeidx, start_point, end_point, savedir=None):
    letterstr = 'abcdefghijklmn'
    proj = crs.PlateCarree()
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          scale='50m',
                                          facecolor='none',
                                          name='admin_1_states_provinces_lines')
    land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                        scale='50m',
                                        facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                         scale='50m',
                                         facecolor=cfeature.COLORS['water'])

    fig = plt.figure(figsize=(24, 18))
    # fig.subplots_adjust(hspace=0.1, wspace=0.05)
    for i, wrf_path in zip(range(0, len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(3, 4, i + 1, projection=proj)
        ax.set_extent(extent, proj)
        ncfile = Dataset(wrf_path)
        # pressure_levels = [300, 400, 500, 700, 850, 900, 925, 950, 975, 1000]
        pressure_levels = np.arange(0.0, 15.0, 0.5)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        ctt = getvar(ncfile, "ctt", timeidx=timeidx)
        slp = getvar(ncfile, "slp", timeidx=timeidx)
        smooth_slp = smooth2d(slp, 3)
        lats, lons = latlon_coords(slp)
        contour_levels = [960, 965, 970, 975, 980, 990]
        c1 = ax.contour(lons, lats, to_np(smooth_slp), levels=contour_levels,
                        colors="white", transform=crs.PlateCarree(), zorder=3,
                        linewidths=1.0)
        contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
        ctt_contours = ax.contourf(to_np(lons), to_np(lats), to_np(ctt),
                                   contour_levels, cmap=get_cmap("Greys"),
                                   transform=crs.PlateCarree(), zorder=2)
        ax.plot([start_point.lon, end_point.lon],
                [start_point.lat, end_point.lat], color="yellow", marker="o",
                transform=crs.PlateCarree(), zorder=3)
        ax.add_feature(land)
        ax.add_feature(states, linewidth=.5, edgecolor="black")
        ax.coastlines('50m', linewidth=1)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks(np.arange(extent[1] + 2, extent[0], 4), crs=proj)
        if i == 0:
            ax.set_yticks(np.arange(extent[2] + 2, extent[3], 4), crs=proj)
        else:
            ax.set_yticks([])
        ax.tick_params(colors='k')

    cb_ax = fig.add_axes([0.92, 0.56, 0.008, 0.3])
    cb_ctt = fig.colorbar(ctt_contours, cax=cb_ax)
    cb_ctt.ax.tick_params(labelsize=12)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
    # fig.tight_layout()

    for i, wrf_path in zip(range(0, len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(3, 4, i + 1+len(wrf_paths))

        ncfile = Dataset(wrf_path)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        wspd = getvar(ncfile, "wspd_wdir", units="kt", timeidx=timeidx)[0, :]
        z_cross = vertcross(wspd, z,  # levels=pressure_levels[::-1],
                            wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        lats, lons = latlon_coords(wspd)

        z_cross = z_cross[0:100, :]
        c = ax.contourf(to_np(z_cross), levels=np.arange(0, 120, 5),
                           cmap=get_cmap('rainbow'))
        coord_pairs = to_np(z_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str(fmt="({:.1f}N, {:.1f}W)") for pair in np.abs(to_np(coord_pairs))]
        ax.set_xticks(x_ticks[::20])
        ax.set_xticklabels(x_labels[::20], fontsize=10)
        # ax.text(0.05, 0.1, s=letterstr[i], fontsize=12, transform=ax.transAxes, zorder=3)
        if i == 0:
            wspd_vals = to_np(z_cross.coords["vertical"])
            wspd_ticks = np.arange(wspd_vals.shape[0])
            ax.set_yticks(wspd_ticks[::20])
            ax.set_yticklabels(["{:.2f}".format(each) for each in wspd_vals[::20]], fontsize=12)
    vt_ax = fig.add_axes([0.92, 0.13, 0.008, 0.3])
    cb_vt = fig.colorbar(c, cax=vt_ax)


    for i, wrf_path in zip(range(0, len(wrf_paths)), wrf_paths):
        ax = fig.add_subplot(3, 4, i + 1+2*len(wrf_paths))

        ncfile = Dataset(wrf_path)
        # Get the WRF variables
        z = getvar(ncfile, "z", units='km', timeidx=timeidx)
        wspd = getvar(ncfile, "temp", units="degC", timeidx=timeidx)
        z_cross = vertcross(wspd, z,  # levels=pressure_levels[::-1],
                            wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        lats, lons = latlon_coords(wspd)

        z_cross = z_cross[0:100, :]
        c = ax.contourf(to_np(z_cross), levels=np.arange(-80, 30, 1),
                           cmap=get_cmap('jet'))
        coord_pairs = to_np(z_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str(fmt="({:.1f}N, {:.1f}W)") for pair in np.abs(to_np(coord_pairs))]
        ax.set_xticks(x_ticks[::20])
        ax.set_xticklabels(x_labels[::20], fontsize=10)
        # ax.text(0.05, 0.1, s=letterstr[i], fontsize=12, transform=ax.transAxes, zorder=3)
        if i == 0:
            wspd_vals = to_np(z_cross.coords["vertical"])
            wspd_ticks = np.arange(wspd_vals.shape[0])
            ax.set_yticks(wspd_ticks[::20])
            ax.set_yticklabels(["{:.2f}".format(each) for each in wspd_vals[::20]], fontsize=12)
    # vt_ax = fig.add_axes([0.92, 0.13, 0.008, 0.3])
    # cb_vt = fig.colorbar(c, cax=vt_ax)

    # plt.subplot_tool()
    # plt.show()


    if savedir == None:
        plt.show()
    if savedir != None:
        if os.path.exists(savedir):
            plt.savefig(os.path.join(savedir, "ctt_vert_wspd_temp_subplot_all_solution_" + str(timeidx) + ".png"), dpi=600,
                        bbox_inches='tight')
        else:
            try:
                os.makedirs(savedir)
                plt.savefig(os.path.join(savedir, "ctt_vert_wspd_temp_subplot_all_resolution_" + str(timeidx) + ".png"), dpi=600,
                            bbox_inches='tight')
            except OSError:
                print("Could not create directory")
    pass
