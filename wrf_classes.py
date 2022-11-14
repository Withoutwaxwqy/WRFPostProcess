# -*- encoding: utf-8 -*-
'''
@File    :   wrf_classes.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/4/2 16:05   WangQiuyi      1.0         None
'''
import datetime as dt


class WrfTimes():
    def __init__(self, stime, step, interval, fmt="%Y-%m-%d_%H_%M_%S"):
        # super().__init__()
        self.fmt = fmt
        self.stime = stime
        self.step = step
        self.interval = interval
        self.etime = stime + dt.timedelta(hours=self.step*self.interval)

    def __getitem__(self, item):
        if 0 <= item < self.step:
            return self.stime + dt.timedelta(hours=item*self.interval)
        else:
            print('out of steps!\n')

    def TimeSeriesFormatOut(self, fmt=None):
        '''
        输出fmt时间序列
        :return:
        '''
        out = []
        stime = self.stime
        out.append(stime.strftime(self.fmt))
        for i in range(self.step+1):
            t = stime + dt.timedelta(hours=self.interval)
            out.append(t.strftime(self.fmt))
            stime = t
        return out

    def TimeSeriesDateTimeListOut(self):
        '''
        生成时间序列
        :return:
        '''
        out = []
        for i in range(self.step + 1):
            t = self.stime + dt.timedelta(hours=self.interval)
            out.append(t)
        return out


