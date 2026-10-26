


# !/usr/bin/env python
# -*- coding:utf-8 -*-


# 将数据写入excel中
from fractions import Fraction
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义基本参数
p_w = {"A": 1/3, "B": 2/3}  # 消息概率
# c_legit = {"U": {"U": 0.9, "R": 0.1}, "R": {"U": 0.2, "R": 0.8}, "∅": {"U": 0.1, "R": 0.1}}  # 合法信道
# c_eaves = {"U": {"U": 0.6, "R": 0.4}, "R": {"U": 0.3, "R": 0.7}, "∅": {"U": 0.2, "R": 0.3}}  # 窃听信道

d_w_w_hat = {"A": {"A": 0, "B": 1}, "B": {"A": 1, "B": 0}}  # 失真矩阵

U = {
    ('A', '∅'): 1/9, ('A', 'U'): 1/9, ('A', 'R'): 1/9, ('A', 'UU'): 1/9, ('A', 'UR'): 1/9, ('A', 'RU'): 1/9, ('A', 'RR'): 0, ('A', 'UUR'): 1/9,
    ('A', 'URU'): 1/9, ('A', 'RUU'): 1/9, ('A', 'URR'): 0, ('A', 'RUR'): 0, ('A', 'RRU'): 0,('A', 'UURR'): 0,
    ('B', '∅'): 1/19, ('B', 'U'): 1/19, ('B', 'R'): 1/19, ('B', 'UU'): 1/19, ('B', 'UR'): 1/19, ('B', 'RU'): 1/19, ('B', 'RR'): 1/19, ('B', 'UUR'): 1/19,
    ('B', 'URU'): 1/19, ('B', 'RUU'): 1/19, ('B', 'URR'): 1/19, ('B', 'RUR'): 1/19, ('B', 'RRU'): 1/19, ('B', 'UURR'): 6/19,
}




# 定义 S 集合
# 所有映射的集合
S = ["∅", "U", "R", "UU", "UR", "RU", "RR", "UUR", "URU", "RUU", "URR", "RUR", "RRU", "UURR"]
# 含义A映射的集合
S_A = ["UUR", "URU", "RUU"]
# 含义B映射的集合
S_B = ["UURR", "URUR", "URRU", "RRUU", "RURU", "RUUR"]

# 初始化边缘概率
P_S1 = {s_hat: 0 for s_hat in S}  # 用于计算 P(S1)
P_S2 = {s_hat: 0 for s_hat in S}  # 用于计算 P(S2)


# 信道转移概率 (合法接收方)
def c(s1, s):
    if s1 == s:
        return 0.8
    elif s1 != s and s1 != '∅':
        return 0.18/12
    elif s1 == '∅':
        return 0.02
    return 0


# 窃听者的信道转移概率
def ca(s2, s):
    if s2 == s:
        return 0.5
    elif s2 != s and s2 != '∅':
        return 0.25/12
    elif s2 == '∅':
        return 0.25
    return 0


# 计算 P(S1) 和 P(S2) 边缘概率
for (w, s), p_s in U.items():
    for s_hat in S:
        P_S1[s_hat] += p_s * c(s_hat, s)
        P_S2[s_hat] += p_s * ca(s_hat, s)


# 含义A的保密性
# 计算互信息
def compute_mutual_information_y(w, s, P_S_hat):
    mutual_info = 0
    # for s in S:  # 对w可能映射的每一个s
    p_s = U[(w, s)]  # 映射的概率
    for s_hat in S:
        joint_prob = p_s * c(s_hat, s)
        if joint_prob > 0 and P_S_hat[s_hat] > 0:
            mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
    return mutual_info  # 每个s对应的互信息值


def compute_mutual_information_z(w, s,P_S_hat):
    mutual_info = 0
    # for s in S:
    p_s = U[(w, s)]
    for s_hat in S:
        joint_prob = p_s * ca(s_hat, s)
        if joint_prob > 0 and P_S_hat[s_hat] > 0:
            mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
    return mutual_info


results = {}
for s in S:
    # 计算失真
    i_x_y_A = compute_mutual_information_y('A', s, P_S1)
    i_x_z_A = compute_mutual_information_z('A', s, P_S2)
    secrecy_A = i_x_y_A - i_x_z_A
    i_x_y_B = compute_mutual_information_y('B', s, P_S1)
    i_x_z_B = compute_mutual_information_z('B', s, P_S2)
    secrecy_B = i_x_y_B - i_x_z_B
    results[s] = {
        "secrecy_A": secrecy_A,
        "secrecy_B": secrecy_B,
    }


for s, values in results.items():
    print(f"A->{s}, B->{s}, secrecy_A: {values['secrecy_A']:.4f}, secrecy_B: {values['secrecy_B']:.4f}")
