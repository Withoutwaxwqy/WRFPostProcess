# -*- encoding: utf-8 -*-
'''
@File    :   pwvcat.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/5/18 17:21   WangQiuyi      1.0         None
'''
import datetime
import datetime as dt


def pwv_6h_cat(time, path):
    pass


def gen_shell_cat(time):
    pattern = "%Y%m%d%H"
    t1 = time + datetime.timedelta(hours=-3)
    t2 = time + datetime.timedelta(hours=-2)
    t3 = time + datetime.timedelta(hours=-1)
    t4 = time
    t5 = time + datetime.timedelta(hours=1)
    t6 = time + datetime.timedelta(hours=2)
    out = "cat obs.{} obs.{} obs.{} obs.{} obs.{} obs.{} > pwv.{}\n"
    outstr = out.format(t1.strftime(pattern), t2.strftime(pattern), t3.strftime(pattern), t4.strftime(pattern),
                        t5.strftime(pattern), t6.strftime(pattern), t4.strftime(pattern))
    print(outstr)


t0 = datetime.datetime(2021, 8, 26, 00)
for i in range(8):
    gen_shell_cat(t0)
    t0 = t0 + datetime.timedelta(hours=6)
