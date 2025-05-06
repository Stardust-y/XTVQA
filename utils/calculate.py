from scipy.stats import entropy

import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
import torch
def compute_entropy(logits):
    entropy_sum = 0.0
    union_prob = np.ones_like(logits[0].cpu().numpy())
    for step_logits in logits:
        # max_logit = np.max(step_logits)
        # step_logits = step_logits - max_logit
        # print("exp", np.exp(step_logits), np.sum(np.exp(step_logits)))
        # probs = np.exp(step_logits) / np.sum(np.exp(step_logits))
        probs = F.softmax(step_logits)
        probs = probs.cpu().numpy().astype(np.float32)

        step_entropy = entropy(probs)
        # # entropy = -np.sum(probs * np.log(probs))
        # # 累加当前时间步的熵
        entropy_sum += step_entropy

    avg_entropy = entropy_sum / logits.shape[0]

    return avg_entropy
def compute_entropy_v2(logits):
    prob_list = []
    union_prob = torch.ones((1,5))
    m = torch.nn.Softmax().cuda()
    for i, step_logits in enumerate(logits[:1]):

        probs = F.softmax(step_logits)
        # probs = probs.cpu().numpy()
        # 对probs进行topp
        top_p = 0.999
        top_k = 3
        topk_probs, topk_indices = torch.topk(probs, k=3, dim=-1)
        # 计算需要保留的最大tokens数量
        # print("probs", probs)
        # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # print("sort", sorted_probs)
        # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # print("cu", cumulative_probs)
        # keep_seq = cumulative_probs <= top_p
        # keep_nums = keep_seq.sum(dim=-1)
        # print(keep_nums)
        #
        # # 使用torch.topk函数进行top-p采样
        # topk_probs, topk_indices = torch.topk(probs, keep_nums.item(), dim=-1)

      #  print(topk_probs, topk_probs.shape)

        # pdb.set_trace()
        if type(union_prob) is np.ndarray:
            union_prob = np.outer(union_prob, topk_probs.cpu()).flatten()
           # print("union", union_prob.shape)
        else:
            union_prob = np.outer(union_prob.cpu(), topk_probs.cpu()).flatten()
        print(type(union_prob))
        # union_prob *= probs
        # print(union_prob)

    # avg_entropy = sum(union_prob_sum)
    #avg_entropy = entropy_sum / logits.shape[0]
    avg_entropy = entropy(union_prob)
    return avg_entropy

def compute_entropy_v3(logits):
    entropy_total = 0.0
    length, _ = logits.shape
    print(length, logits.shape)
    for i, step_logits in enumerate(logits):

        probs = F.softmax(step_logits)

        entropy_total += entropy(probs.cpu().numpy())
    entropy_total = entropy_total / length

    return entropy_total
def compute_mi(logits, logits_wo):
    """
    计算两个 logits 分布 p 和 q 之间的互信息
    """
    # logits = logits.cpu().numpy().astype(np.float32)
    # logits_wo = logits_wo.cpu().numpy().astype(np.float32)
    avg_entropy = compute_entropy_v3(logits)
    avg_entropy_wo = compute_entropy_v3(logits_wo)
    print("entropy", avg_entropy_wo, avg_entropy)
    mi = avg_entropy_wo - avg_entropy

    return mi, avg_entropy_wo, avg_entropy

# def compute_pmi(logits, logits_wo, target_ids):
#     cut_index = min(min(target_ids.shape[1], logits.shape[0]), logits_wo.shape[0])
#     print("cut index", cut_index, target_ids.shape)
#     logits = logits[:cut_index]
#     logits_wo = logits_wo[:cut_index]
#     print("before", target_ids.shape)
#     target_ids = target_ids[:,:cut_index]
#     print("after", target_ids.shape)
#
#     print("shaoe", logits.view(-1, logits.size(-1)).shape, logits_wo.view(-1, logits_wo.size(-1)).shape, target_ids.view(-1).shape)
#     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1)).float().item()
#     loss_wo = F.cross_entropy(logits_wo.view(-1, logits_wo.size(-1)), target_ids.view(-1)).float().item()
#
#     print(loss, loss_wo)
#     return loss_wo - loss, loss, loss_wo
    # selected_values = logits.index_select(0, torch.tensor(gt_ids))
    # selected_values_wo = logits_wo.index_select(0, torch.tensor(gt_ids))
def compute_pmi(logits, logits_wo, target_ids):
    print("shapes", logits.shape, logits_wo.shape, target_ids.shape)
    cut_index = min(min(target_ids.shape[1], logits.shape[0]), logits_wo.shape[0])
    _, e, e_wo = compute_mi(logits, logits_wo)

    logits = logits[:cut_index]
    logits_wo = logits_wo[:cut_index]
    target_ids = target_ids[:,:cut_index]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1)).float().item()
    loss_wo = F.cross_entropy(logits_wo.view(-1, logits_wo.size(-1)), target_ids.view(-1)).float().item()

    print(loss, loss_wo)
    return loss_wo - loss, loss, loss_wo, e, e_wo