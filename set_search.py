

# !/usr/bin/env python
# -*- coding:utf-8 -*-


# 将数据写入excel中
from fractions import Fraction
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import for_d_no_s


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


# 定义符号长度
symbol_lengths = {"U": 2, "R": 1, "∅": 0}


# 计算符号的总长度
def compute_cost(s):
    return sum(symbol_lengths[char] for char in s)


# 计算字符串s作为前缀在word_set中出现的次数
def count_prefix_occurrences(s, word_set):
    return sum(s == word[: len(s)] for word in word_set)


# 计算q（w|s)
def calculate_probabilities(s, S_A, S_B):
    if s == '∅':  # 空字符串作为前缀在任何字符串中都出现，但这里我们不考虑它
        prob_A_given_s = Fraction(1, 3)
        prob_B_given_s = Fraction(2, 3)
    occurrences_A = count_prefix_occurrences(s, S_A)
    occurrences_B = count_prefix_occurrences(s, S_B)
    total_occurrences = occurrences_A + occurrences_B

    if total_occurrences == 0:
        return Fraction(1, 3), Fraction(2, 3)  # 如果s不在任何集合中作为前缀出现

    prob_A_given_s = Fraction(occurrences_A, total_occurrences)
    prob_B_given_s = Fraction(occurrences_B, total_occurrences)
    return prob_A_given_s, prob_B_given_s


results_q = {s: calculate_probabilities(s, S_A, S_B) for s in S}


# 其实是phi
# 计算失真

def compute_distortion(w, s):
    if w == 'A':
        q_val = results_q[s][0]  # 求q
    else:
        q_val = results_q[s][1]

    return 1 - q_val


# 计算 P(S1) 和 P(S2) 边缘概率
for (w, s), p_s in U.items():
    for s_hat in S:
        P_S1[s_hat] += p_s * c(s_hat, s)
        P_S2[s_hat] += p_s * ca(s_hat, s)


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


def discover_subsets(w, S, compute_distortion, compute_cost):
    """
    Discover the six subsets of messages for a meaning w.

    Parameters:
        w: the meaning.
        S: the set of messages.
        phi: a function phi(w, s) providing semantic distortion.
        ell: a function ell(s) providing cost for message s.

    Returns:
        S_w: Dictionary containing the six subsets {S'_w, S''_w, S'_w_s, S''_w_s, S'_w_s', S''_w_s'}.
    """
    # Initialize subsets
    S_prime_low_w = S.copy()  # s'_w
    S_double_prime_low_w = S.copy()  # s''_w

    s_low_l = S[0]  #
    s_low_r = S[-1]

    S_prime_up_w = S.copy()  #
    S_double_prime_up_w = S.copy()  # Start with s_M

    s_up_l = S[0]
    s_up_r = S[-1]

    M = len(S)
    for m in range(2, len(S) + 1):# 0-M-1
        sm = S[m - 1]
        if compute_distortion(w, s_low_l) <= compute_distortion(w, sm):
            #S_prime_low_w -= {sm}
            S_prime_low_w.remove(sm)
        elif compute_cost(s_low_l) == compute_cost(sm):
            #S_prime_low_w remove(s_low_l)
            S_prime_low_w.remove(s_low_l)
            s_low_l = sm
        else:
            s_low_l = sm

        sm_plus = S[M-m]
        if compute_distortion(w, s_low_r) <= compute_distortion(w, sm_plus):
            #S_double_prime_low_w.remove(sm_plus)
            S_double_prime_low_w.remove(sm_plus)
        elif compute_cost(s_low_r) == compute_cost(sm_plus):
            #S_double_prime_low_w.remove(s_low_r)
            S_double_prime_low_w.remove(s_low_r)
            s_low_r = sm_plus
        else:
            s_low_r = sm_plus

        # sm_minus = S[m - 1]
        if compute_distortion(w, s_up_l) >= compute_distortion(w, sm):
            #S_prime_up_w -= {sm}
            S_prime_up_w.remove(sm)
        elif compute_cost(s_up_l) == compute_cost(sm):
            #S_prime_up_w.remove(s_up_l)
            S_prime_up_w.remove(s_up_l)
            s_up_l = sm
        else:
            s_up_l = sm

        if compute_distortion(w, s_up_r) >= compute_distortion(w, sm_plus) :
            #S_double_prime_up_w.remove(sm_plus)
            S_double_prime_up_w.remove(sm_plus)
        elif compute_cost(s_up_r) == compute_cost(sm_plus):
            #S_double_prime_up_w.remove(s_up_r)
            S_double_prime_up_w.remove(s_up_r)
            s_up_r = sm_plus
        else:
            s_up_r = sm_plus

    S_low_w = {
        "S_prime_low_w": S_prime_low_w,
        "S_double_prime_low_w": S_double_prime_low_w,
        "S_low_w": S_prime_low_w + S_double_prime_low_w
    }

    S_up_w ={
        "S_prime_up_w":S_prime_up_w,
        "S_double_prime_up_w":S_double_prime_up_w,
        "S_up_w": S_prime_up_w+S_double_prime_up_w
    }

    return S_low_w["S_low_w"], S_up_w["S_up_w"]



# Discover subsets
S_low_w_A, S_up_w_A= discover_subsets("A", S, compute_distortion, compute_cost)
S_low_w_B, S_up_w_B= discover_subsets("B", S, compute_distortion, compute_cost)

print(S_low_w_A, S_up_w_A)
# print(S_low_w_A+ S_up_w_A)
print(S_low_w_B, S_up_w_B)
# print(S_low_w_B+ S_up_w_B)
