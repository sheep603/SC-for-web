from fractions import Fraction
import itertools
import numpy as np

# 定义基本参数
p_w = {"A": 1/3, "B": 2/3}  # 含义概率
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
# 构建所有编码方案
encoding_schemes = [
    {"A": a, "B": b} for a, b in itertools.product(S, S)
]

# 初始化边缘概率
P_S1 = {s_hat: 0 for s_hat in S}  # 用于计算 P(S1)
P_S2 = {s_hat: 0 for s_hat in S}  # 用于计算 P(S2)

# 定义符号长度
symbol_lengths = {"U": 2, "R": 1, "∅": 0}


# 计算符号的总长度
def compute_cost(s):
    return sum(symbol_lengths[char] for char in s)


# 成本计算
def compute_cost_total(encoding, p_w):
    cost = 0
    for w in p_w:
        for s in encoding[w]:
            cost += p_w[w] * compute_cost(s)
    return cost


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


# 计算失真--p_channel上进行修改

# def compute_distortion(encoding, p_w, S, p_channel):
#     """
#
#     :param encoding: 编码方案
#     :param p_w: 含义的概率
#     :param S:所有的消息集合
#     :param p_channel:信道转移概率
#     :return:
#     """
#     distortion = 0
#     for w in p_w:
#         s = encoding[w]
#         for s_prime in S:
#             for w_hat in p_w:
#                 if w == 'A':
#                     index_w = 0
#                 else:
#                     index_w = 1
#                 q_val = results_q[s_prime][index_w]  # 求q
#                 # p(s_prime,s)需要跟下面的修改一下
#                 distortion += p_w[w] * p_channel * q_val * d_w_w_hat[w][w_hat]
#     return distortion


# 计算失真--p_channel上进行修改
def compute_distortion(w,p_w, s, s_1, p_channel):
    """
    :param p_w: 含义的概率
    :param s:接收的消息
    :param p_channel:信道转移概率
    :return:
    """
    for w_hat in p_w:
        if w == 'A':
            index_w = 0
        else:
            index_w = 1
        q_val = results_q[s_1][index_w]  # 求q
        # p(s_prime,s)需要跟下面的修改一下
        distortion = p_w[w] * p_channel * q_val * d_w_w_hat[w][w_hat]
    return distortion


# 计算 P(S1) 和 P(S2) 边缘概率
def cal_side(p_channel1, p_channel2, U):
    for (w, s), p_s in U.items():
        for s_hat in S:
            P_S1[s_hat] += p_s * p_channel1 #c(s_hat, s)信道转移概率
            P_S2[s_hat] += p_s * p_channel2


# 计算互信息--s-s1
# def compute_mutual_information_y(encoding, p_w, S,P_S_hat,p_channel):
#     mutual_info = 0
#     for w in p_w:
#         for s in encoding[w]:
#             p_s = U[(w, s)]
#             for s_hat in S:
#                 joint_prob = p_s * p_channel
#                 if joint_prob > 0 and P_S_hat[s_hat] > 0:
#                     mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
#     return mutual_info

# 计算互信息--s-s1

def compute_mutual_information_y(w, s, s_1, P_S_hat ,p_channel):
    mutual_info = 0
    p_s = U[(w, s)]
    joint_prob = p_s * p_channel
    if joint_prob > 0 and P_S_hat[s_1] > 0:
        mutual_info = joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_1]))
    return mutual_info


# 计算s-s2
# def compute_mutual_information_z(encoding, p_w, S,P_S_hat,p_channel):
#     mutual_info = 0
#     for w in p_w:
#         for s in encoding[w]:
#             p_s = U[(w, s)]
#             for s_hat in S:
#                 joint_prob = p_s * p_channel
#                 if joint_prob > 0 and P_S_hat[s_hat] > 0:
#                     mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_hat]))
#     return mutual_info


def compute_mutual_information_z(w,s, s_2,P_S_hat,p_channel):
    mutual_info = 0
    p_s = U[(w, s)]
    joint_prob = p_s * p_channel
    if joint_prob > 0 and P_S_hat[s_2] > 0:
        mutual_info += joint_prob * np.log2(joint_prob / (p_s * P_S_hat[s_2]))
    return mutual_info


# S转换成二进制串
def string_to_binary(s):
    if not s:
        return ''
    # 使用字典将 'U' 映射为 0，'R' 映射为 10
    mapping = {'U': '0', 'R': '10', '∅': '11'}
    # 对字符串中的每个字符进行映射，返回对应的二进制串
    return ''.join([mapping[char] for char in s])


# 码本
codebook = {message: string_to_binary(message) for message in S}


# 最大似然解码--得到解码后的message
def maximum_likelihood_decoding(received_signal, codebook):
    min_distance = float('inf')  # 初始化最小距离
    decoded_message = None  # 最优消息
    received_length = len(received_signal)

    # 遍历码本中的每个消息及其对应的编码信号
    for message, encoded_signal in codebook.items():
        # 将二进制编码转为数值序列
        numeric_signal = np.array([int(bit) for bit in encoded_signal])
        # 计算接收信号与编码信号之间的欧氏距离
        if len(numeric_signal) == received_length:
            distance = np.sum((received_signal - numeric_signal) ** 2)
        # 更新最优解
            if distance < min_distance:
                min_distance = distance
                decoded_message = message
    return decoded_message


# 模拟AWGN信道---生成接收的信号
def awgn_channel(s, snr_db):
    """
    s: 发送的信息
    snr_db: 信噪比
    """
    snr = 10 ** (snr_db / 10)  # 将信噪比从dB转换为线性
    if isinstance(s, str):
        s = np.array([int(char) for char in s])
    power_signal = np.mean(s ** 2)  # 计算信号的功率
    noise_power = power_signal / snr  # 根据信噪比计算噪声功率
    noise = np.random.normal(0, np.sqrt(noise_power), size=s.shape)  # 高斯噪声
    return s + noise  # 返回接加入噪声的信号


# 信道转移概率计算 (高斯信道)
def p_s_given_s(s_prime, s, sigma):
    """
    计算高斯信道下，接收信号 s' 给定发送信号 s 的转移概率 p(s'|s)
    sigma: 高斯噪声的标准差
    """
    s_prime = np.asarray(s_prime, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    p = (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-((s_prime - s) ** 2) / (2 * sigma ** 2))
    return np.clip(p, 0, 1)


# 仿真过程
def simulate_channel(x, snr_db_R=10, snr_db_T=2, epsilon=1e-6):
    """
    仿真信号的传输过程，并计算保密性与失真
    snr_db_R: 合法接收方的信噪比（dB）
    snr_db_T: 窃听者的信噪比（dB）
    sigma: 高斯噪声标准差
    :return: 失真与保密性值
    """
    # x是发送的信号
    # x = codebook[s]
    # 通过AWGN信道模拟接收的信号
    y_R = awgn_channel(x, snr_db_R)  # 合法接收方接收的信号
    y_T = awgn_channel(x, snr_db_T)  # 窃听者接收的信号
    # print(x, y_R, y_T)
    # 合法接收方解码（最大似然解码）解码的消息
    s_1 = maximum_likelihood_decoding(y_R, codebook)
    # print(s_1)
    # 窃听者解码（最大似然解码）
    s_2 = maximum_likelihood_decoding(y_T, codebook)
    if isinstance(x, str):
        x = np.array([int(char) for char in x])
    sigma_R = max(epsilon,np.sqrt(np.mean(x ** 2) / (10 ** (snr_db_R / 10))))
    sigma_T = max(epsilon,np.sqrt(np.mean(x ** 2) / (10 ** (snr_db_T / 10))))
    print(sigma_R, sigma_T)

    # print(s_2)
    # 现在发送的是s,接受的s1,s2
    # 计算信道转移概率 p(s'|s)：返回的是每个二进制信号位的转移概率
    p_s_given_s_1 = np.array([p_s_given_s(y_R[i], x[i], sigma_R) for i in range(len(x))])
    p_s_given_s_2 = np.array([p_s_given_s(y_T[i], x[i], sigma_T) for i in range(len(x))])
    # print(p_s_given_s_1, p_s_given_s_2)
    # 忽略零转移概率符号
    non_zero_R = p_s_given_s_1[p_s_given_s_1 > 0]
    non_zero_T = p_s_given_s_2[p_s_given_s_2 > 0]
    # 信息转移概率
    p_s_s1 = np.prod(non_zero_R)
    p_s_s2 = np.prod(non_zero_T)
    # 返回解码的信息，以及转移概率
    return s_1, s_2, p_s_s1, p_s_s2


# 7. 运行仿真
results = {}
for scheme in encoding_schemes:
    # 计算失真和互信息
    distortion = 0
    i_x_y = 0
    i_x_z = 0
    secrecy = 0
    for w in p_w:
        s = scheme[w]
        x = codebook[s]
        s_1, s_2, p_s_s1, p_s_s2 = simulate_channel(x, snr_db_R=10, snr_db_T=2,epsilon=1e-6)
        # print(w,s,s_1,s_2,p_s_s1,p_s_s2)
        #compute_distortion(w,p_w, s, s_1, p_channel):
        cal_side(p_s_s1, p_s_s2, U)
        distortion += compute_distortion(w, p_w, s, s_1, p_s_s1)
        # def compute_mutual_information_y(w, s, p_w, s_1, P_S_hat, p_channel):
        i_x_y += compute_mutual_information_y(w, s, s_1, P_S1, p_s_s1)
        i_x_z += compute_mutual_information_z(w, s, s_2, P_S2, p_s_s2)
    secrecy = -(i_x_y - i_x_z)
    cost = compute_cost_total(scheme, p_w)
    results[scheme["A"], scheme["B"]] = {
        "Distortion": distortion,
        "Secrecy": secrecy,
        "Cost": cost
    }
for (a, b), values in results.items():
    print(f"Scheme: A->{a}, B->{b}, Distortion: {values['Distortion']:.4f}, Secrecy: {values['Secrecy']:.4f}, Cost: {values['Cost']:.4f}")


