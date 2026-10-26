

# !/usr/bin/env python
# -*- coding:utf-8 -*-

from fractions import Fraction
import itertools
import numpy as np


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


# 所有结果
results_q = {s: calculate_probabilities(s, S_A, S_B) for s in S}

# 构建所有编码方案
encoding_schemes = [
    {"A": a, "B": b} for a, b in itertools.product(S, S)
]


# 成本计算
def compute_cost_total(encoding, p_w):
    cost = 0
    for w in p_w:
        for s in encoding[w]:
            cost += p_w[w] * compute_cost(s)
    return cost


# 计算失真

def compute_distortion(encoding, p_w,S):
    distortion = 0
    for w in p_w:
        s = encoding[w]
        for s_prime in S:
            for w_hat in p_w:
                if w == 'A':
                    index_w = 0
                else:
                    index_w = 1
                q_val = results_q[s][index_w]  # 求q
                # print(p_w[w])
                # print(c(s_prime, s))
                # print(q_val)
                # print(d_w_w_hat[w][w_hat])
                # print(U[(w, s)])
                #distortion += p_w[w] * U[(w, s)] * c(s_prime, s) * q_val * d_w_w_hat[w][w_hat]
                distortion += p_w[w] * c(s_prime, s) * q_val * d_w_w_hat[w][w_hat]
    return distortion


# 计算 P(S1) 和 P(S2) 边缘概率
for (w, s), p_s in U.items():
    for s_hat in S:
        P_S1[s_hat] += p_s * c(s_hat, s)
        P_S2[s_hat] += p_s * ca(s_hat, s)


# # 计算互信息公式
#     def mutual_information(P_S_hat):
#         mutual_info = 0
#         for (w, s), p_s in U.items():
#             for s_hat in S:
#                 joint_prob = p_s * c(s_hat, s)
#                 if joint_prob > 0 and P_S_hat[s_hat] > 0:  # 避免 log(0)
#                     mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
#         return mutual_info
#
#
#     I_S1 = mutual_information(P_S1)
#     I_S2 = mutual_information(P_S2)
#     secrecy = I_S1 - I_S2

# 计算互信息
def compute_mutual_information_y(encoding, p_w, S,P_S_hat):
    mutual_info = 0
    for w in p_w:
        for s in encoding[w]:
            p_s = U[(w, s)]
            for s_hat in S:
                joint_prob = p_s * c(s_hat, s)
                if joint_prob > 0 and P_S_hat[s_hat] > 0:
                    mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
    return mutual_info


def compute_mutual_information_z(encoding, p_w, S,P_S_hat):
    mutual_info = 0
    for w in p_w:
        for s in encoding[w]:
            p_s = U[(w, s)]
            for s_hat in S:
                joint_prob = p_s * ca(s_hat, s)
                if joint_prob > 0 and P_S_hat[s_hat] > 0:
                    mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
    return mutual_info


# 遍历所有编码方案
results = {}
for scheme in encoding_schemes:
    # 计算失真
    distortion = compute_distortion(scheme, p_w, S)
    i_x_y = compute_mutual_information_y(scheme, p_w, S, P_S1)
    i_x_z = compute_mutual_information_z(scheme, p_w, S, P_S2)
    secrecy = i_x_y - i_x_z
    cost = compute_cost_total(scheme, p_w)
    results[scheme["A"], scheme["B"]] = {
        "Distortion": distortion,
        "Secrecy": secrecy,
        "Cost": cost
    }

# 输出结果
for (a, b), values in results.items():
    print(f"Scheme: A->{a}, B->{b}, Distortion: {values['Distortion']:.4f}, Secrecy: {values['Secrecy']:.4f}, Cost: {values['Cost']:.4f}")

