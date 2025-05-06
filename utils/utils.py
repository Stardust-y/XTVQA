import requests
import re, logging
#import urlib
from rouge import Rouge
import numpy as np
from sklearn.neighbors import KernelDensity
def calc_rouge(hypotheses, references):


    rouge = Rouge()
    rouge_scores = {
        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
    }

    for hyp, ref in zip(hypotheses, references):
        if len(hyp) == 0 or len(ref) == 0:
            continue
        scores = rouge.get_scores(hyp, ref)
        scores = scores[0]
        rouge_scores['rouge-1']['f'] += scores['rouge-1']['f']
        rouge_scores['rouge-1']['p'] += scores['rouge-1']['p']
        rouge_scores['rouge-1']['r'] += scores['rouge-1']['r']
        rouge_scores['rouge-2']['f'] += scores['rouge-2']['f']
        rouge_scores['rouge-2']['p'] += scores['rouge-2']['p']
        rouge_scores['rouge-2']['r'] += scores['rouge-2']['r']
        rouge_scores['rouge-l']['f'] += scores['rouge-l']['f']
        rouge_scores['rouge-l']['p'] += scores['rouge-l']['p']
        rouge_scores['rouge-l']['r'] += scores['rouge-l']['r']

    data_count = len(hypotheses)
    for k, v in rouge_scores.items():
        v['f'] /= data_count
        v['p'] /= data_count
        v['r'] /= data_count
    print(rouge_scores)
    return rouge_scores

def new_ip(api):
    new_d_ip = ""

    # 发送请求并通过代理转发
    ip_request = requests.get(api)
    if ip_request.status_code == 200:
        new_d_ip = ip_request.text.split('\n')[0]
        print('new ip:%s' % new_d_ip)
        if re.match("{\"code", new_d_ip):
            logging.error('ip request error')
            new_d_ip = ""
    else:
        logging.error('ip request error')
    return new_d_ip


import numpy as np
from sklearn.neighbors import KernelDensity

import numpy as np
from scipy.stats import entropy

def compute_mi(p, q):
    """
    计算两个 logits 分布 p 和 q 之间的互信息
    """
    eps = 1e-8  # 避免取对数时出现零

    # 估计边缘分布
    p = p + eps
    q = q + eps
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)

    # 估计联合分布
    p_q = np.outer(p_norm, q_norm)
    p_q = p_q / np.sum(p_q)

    # 计算边缘熵
    h_p = entropy(p_norm)
    h_q = entropy(q_norm)

    # 计算联合熵
    h_p_q = entropy(p_q.flatten())

    # 计算互信息
    mi = h_p + h_q - h_p_q

    return mi

def compute_conditional_mi(x, q, p_a_given_x_q):
    """
    计算条件互信息 I(X;A|Q)
    x: 图像特征向量
    q: 问题特征向量
    p_a_given_x_q: 条件概率分布 p(a|x,q)
    """

    # 估计 p(x|q)
    kde_x_given_q = KernelDensity(kernel='gaussian').fit(np.c_[x, q])
    log_p_x_given_q = kde_x_given_q.score_samples(np.c_[x, q])

    # 估计 p(x, a|q)
    kde_x_a_given_q = KernelDensity(kernel='gaussian', weights=p_a_given_x_q).fit(np.c_[x, q])
    log_p_x_a_given_q = kde_x_a_given_q.score_samples(np.c_[x, q])

    # 计算条件熵和互信息
    h_x_given_q = -log_p_x_given_q
    h_x_given_a_q = -(p_a_given_x_q * log_p_x_a_given_q).sum()
    conditional_mi = h_x_given_q - h_x_given_a_q

    return conditional_mi


