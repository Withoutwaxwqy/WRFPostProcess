# -*- encoding: utf-8 -*-
'''
@File    :   downloadRDA.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/5/17 11:28   WangQiuyi      1.0         None
'''
import datetime


def gen_NCAR_download_links(stat, cycletime, interval, forecastime, datasetname):
    """
    生成NCAR数据库的下载链接
    :param stat:
    :param cycletime:
    :param interval:
    :param forecastime:
    :param datasetname:
    :return:
    """

    if datasetname == 'ds084.1':
        # 2021/20210827/gfs.0p25.2021082700.f081.grib2
        out = 'set filelist = ( \\\n'
        time = stat - datetime.timedelta(hours=6)
        demo = "{}/{}/gfs.0p25.{}.f{}.grib2\\\n"
        for i in range(cycletime):
            for k in range(0 + 6, interval + 6 + 1, 6):
                out = out + demo.format(time.strftime("%Y"),
                                        time.strftime("%Y%m%d"),
                                        time.strftime("%Y%m%d%H"),
                                        "{:0=3d}".format(k))
            time = time + datetime.timedelta(hours=interval)
        # 最后一次预报的时间长度和循环的时间间隔是不同的因此需要额外进行一次

        for k in range(0 + 6, forecastime + 6, 6):
            out = out + demo.format(time.strftime("%Y"),
                                    time.strftime("%Y%m%d"),
                                    time.strftime("%Y%m%d%H"),
                                    "{:0=3d}".format(k))
        out = out + ')'
    return out


o = gen_NCAR_download_links(datetime.datetime(2021, 8, 27, 12),
                            6, 6, 72, "ds084.1")
