# -*- encoding: utf-8 -*-
'''
@File    :   PrepScore.py.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time        @Author    @Version    @Desciption
------------       -------     --------    -----------
2022/10/26 17:04   WangQiuyi      1.0         None
'''

# _*_ coding: utf-8 _*_
# @Time    : 2021/10/18 下午3:17
# @Author  : yuxin Zheng
# @File    : PrecipScores.py

import numpy as np


# TN: true negative （真阴性）, true = 0 and predict = 0
# TP: true positive （真阳性）, true = 1 and predict = 1
# FP: false positive （假阳性）, true = 0 and predict = 1
# FN: false negative （假阴性）, true = 1 and predict = 0


def prep_clf(obs, pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=0.1):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold=0.1):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)


def FSC(obs, pre, threshold=0.1):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))


def TS(obs, pre, threshold=0.1):
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits / (hits + falsealarms + misses)


def ETS(obs, pre, threshold=0.1):
    '''
    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
    radar-derived precipitation with model-derived winds.
    Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: ETS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return ETS


def FAR(obs, pre, threshold=0.1):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms)
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)


def MAR(obs, pre, threshold=0.1):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return misses / (hits + misses)


def POD(obs, pre, threshold=0.1):
    '''
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return hits / (hits + misses)


def BIAS(obs, pre, threshold=0.1):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses)


def HSS(obs, pre, threshold=0.1):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))

    return HSS_num / HSS_den


def BSS(obs, pre, threshold=0.1):
    '''
    BSS - Brier skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: BSS value
    '''
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))


def MAE(obs, pre):
    """
    Mean absolute error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()

    return np.mean(np.abs(pre - obs))


def RMSE(obs, pre):
    """
    Root mean squared error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: root mean squared error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))


def PRECIPITION_SKILL_SCORE(obs, pre, mode, threshold=0.1):
    if mode == "ACC":
        return ACC(obs, pre, threshold=threshold)
    if mode == "FSC":
        return FSC(obs, pre, threshold=threshold)
    if mode == "TS":
        return TS(obs, pre, threshold=threshold)
    if mode == "ETS":
        return ETS(obs, pre, threshold=threshold)
    if mode == "FAR":
        return FAR(obs, pre, threshold=threshold)
    if mode == "MAR":
        return MAR(obs, pre, threshold=threshold)
    if mode == "POD":
        return POD(obs, pre, threshold=threshold)
    if mode == "BIAS":
        return BIAS(obs, pre, threshold=threshold)
    if mode == "HSS":
        return HSS(obs, pre, threshold=threshold)
    if mode == "BSS":
        return BSS(obs, pre, threshold=threshold)
    if mode == "MAE":
        return MAE(obs, pre)
    if mode == "RMSE":
        return RMSE(obs, pre)

def score_dataframe_out(csv_dir, solutions, threshold=0.1):
    pass
