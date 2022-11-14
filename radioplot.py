# -*- encoding: utf-8 -*-
'''
@File    :   radioplot.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/7/8 9:36   WangQiuyi      1.0         None
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'  # 将字体设置为黑体'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

labels = np.array(["Alice", "Bob", "Candy", "David", "E", "F"])
dataLenth = len(labels)  # 数据长度
data = np.array([83, 61, 95, 67, 76, 88])

data1 = np.array([69, 21, 93, 90, 100, 22])
angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)  # 根据数据长度平均分割圆周长

# 闭合
data = np.concatenate((data, [data[0]]))
data1 = np.concatenate((data1, [data1[0]]))
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))  # 对labels进行封闭

fig = plt.figure(facecolor="white")  # facecolor 设置框体的颜色
plt.subplot(111, polar=True)  # 将图分成1行1列，画出位置1的图；设置图形为极坐标图
p1=plt.plot(angles, data,
         'b--',
         dashes=(10, 10),
         color='g',
         linewidth=0.5,
         label="label A")
plt.fill(angles, data, facecolor='g', alpha=0.15)

plt.subplot(111, polar=True)  # 将图分成1行1列，画出位置1的图；设置图形为极坐标图
p2=plt.plot(angles, data1,
         'b--',
         dashes=(10, 10),
         color='b',
         linewidth=0.5,
         label="label B")
plt.fill(angles, data1, facecolor='b', alpha=0.15)

# 填充两条线之间的色彩，alpha为透明度

plt.legend(loc='best')
plt.thetagrids(angles * 180 / np.pi, labels)  # 做标签
# plt.figtext(0.52,0.95,'雷达图',ha='center')   #添加雷达图标题
plt.grid(True)
plt.show()
