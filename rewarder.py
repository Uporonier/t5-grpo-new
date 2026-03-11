import random
from typing import List

import torch
import math

class RewardScorer:
    def __init__(self, encoded_key_to_original_docid, original_to_encoded_list, gamma=0.9, rank_db=None):
        """
        Args:
            encoded_key_to_original_docid: 编码串 -> 原始ID (用于解码生成的docid)
            original_to_encoded_list: 原始ID -> Token ID列表 (用于验证Token前缀正确性)
            gamma: 时间步衰减因子
        """
        self.encoded_key_to_original_docid = encoded_key_to_original_docid
        self.original_to_encoded_list = original_to_encoded_list
        self.gamma = gamma
        self.rank_db = rank_db
    
    def _get_local_ranks_for_sequence(self, qid: str, gen_ids: List[int]) -> List[int]:
        """
        核心封装：输入 Query ID 和生成序列，返回每个 Token 对应的排名列表。
        """
        query_prefix_map = self.rank_db.get(qid, {})
        sample_local_ranks = []
        current_prefix_ids = []
        is_off_track = False

        # 定义特殊 ID（建议在 __init__ 中保存，这里演示假设为 0 和 1）
        pad_id = 0
        eos_id = 1

        for token in gen_ids:
            # 1. 处理特殊 Token 或 已偏离路径的情况
            if is_off_track or token == pad_id or token == eos_id:
                # 如果已经偏离，后续全给惩罚分 101；如果是 EOS/PAD，给基础分 1
                sample_local_ranks.append(101 if is_off_track else 1)
                continue

            # 2. 构造当前前缀的 Key
            prefix_key = ",".join(map(str, current_prefix_ids))
            if prefix_key != "":
                prefix_key += ",1"  # 适配你数据库中带逗号的 key 格式

            # 3. 查表获取下一跳的排名
            next_token_ranks = query_prefix_map.get(prefix_key, {})
            
            if token in next_token_ranks:
                rank = next_token_ranks[token]
                sample_local_ranks.append(rank)
                current_prefix_ids.append(token)
            else:
                # 发现不在 Trie 树路径上，标记为“偏离”
                sample_local_ranks.append(101)
                is_off_track = True

        return sample_local_ranks


    def reward_function(self, prompts, completions, completion_ids, **kwargs):
            # 初始化一个列表来存所有样本的 token 奖励
            # 结构: List[List[float]]
            batch_token_rewards = []
            
            ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))

            top_100_docids_batch = kwargs.get("top_100_docids", [[]] * len(completion_ids))
            local_ranks_batch = kwargs.get("local_ranks", [[1]*len(c) for c in completion_ids])
            rank_db = self.rank_db # 假设你把 rank_db 传进来了

            qids = kwargs.get("qid", [])

            for i in range(len(completion_ids)):
                gen_ids = completion_ids[i] # Tensor or List
                local_ranks = local_ranks_batch[i]
                relevant_set = ground_truth_sets[i]
                top_100_list = top_100_docids_batch[i]


                qid = str(qids[i])
                query_prefix_map = rank_db.get(qid, {}) # 获取当前 query 的候选图
                




                # 初始化当前样本的奖励序列，全为0
                # 注意：长度要和 gen_ids 一致
                seq_rewards = [0.0] * len(gen_ids)
                # seq_rewards[0]  = 0.0 # 第一个 token 通常是 pad 或 bos，不需要奖励
                start_idx = 0
                # end_idx = 25 # 不包括
                end_idx = len(gen_ids)
                content_ids = gen_ids[start_idx:end_idx]



                key_str = ",".join(map(str, content_ids))
                
                decoded_docid = self.encoded_key_to_original_docid.get(key_str)

                # --- 2. 计算全局奖励 (GT & Sim) ---
                r_global = 0.0
                # Part A: GT
                if decoded_docid and decoded_docid in relevant_set:
                    r_global += 2

                # print(f"hit_count: {self.hit_count}, total_count: {self.total_count}, hit_rate: {self.hit_count/self.total_count:.4f}")

                decision_weights = []

                # decision_weights.append(0.0) # 第一个 token 不计权重
                 # --- 第一遍扫描：计算总决策权重 ---
                if(r_global > 0):
                    current_prefix_ids = []
                    for t in range(start_idx, end_idx):
                        prefix_key = ",".join(map(str, current_prefix_ids))
                        if prefix_key!= "":
                            prefix_key = prefix_key + ",1"
                    
                        candidates = query_prefix_map.get(prefix_key, {})
                        num_c = len(candidates)
                        
                        # 使用 log1p(num) 作为该步骤的权重，num=1时权重为 log(2)≈0.69
                        # 这样既保证了唯一路径有信号，又让分叉口权重更高
                        weight = math.log1p(num_c) 
                        decision_weights.append(weight)
                        current_prefix_ids.append(gen_ids[t])
                    
                    total_weight = sum(decision_weights)


                # # 动态前缀追踪
                current_prefix_ids = []
                for t in range(start_idx, end_idx -1 ):   #end_idx -1 是因为最后一个 token 是 eos 不计奖励
                    rank = local_ranks[t]
                     # 获取当前前缀下的候选
                    prefix_key = ",".join(map(str, current_prefix_ids))
                    if prefix_key!= "":
                        prefix_key = prefix_key + ","
                    candidates = query_prefix_map.get(prefix_key, {})
                    num_candidates = len(candidates)
                    
                    step_r = 0.0
                    if rank >= 100:
                        step_r = -0.1 #  只给当前走错的这一步一个惩罚
                        break 
                    elif num_candidates == 1:
                        
                        step_r = 0.1  # 只有一条路径可以走 给一个基础奖励
                    else:
                        step_r = 1.0 / math.log(rank+2)    #做出选择的奖励
                    if r_global > 0:
                        seq_rewards[t] = step_r +r_global * (decision_weights[t] / total_weight)
                    else:
                        seq_rewards[t] = step_r

                batch_token_rewards.append(seq_rewards)

            return batch_token_rewards


# 1. 消融实验 A：w/o Branching Weight逻辑：保留 Stepwise 奖励，但 Global 奖励不再根据“分支难度”分配，而是平均分配给每一个 Token（即设权重 $w_t=1$）。
    def reward_function_no_branching(self, prompts, completions, completion_ids, **kwargs):

        """
        [消融实验 1] w/o Branching Weight
        - 局部奖励 (Stepwise): 正常计算 (1/log(rank))
        - 全局奖励分配: 平均分配 (权重恒为 1.0)
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        local_ranks_batch = kwargs.get("local_ranks", [[1]*len(c) for c in completion_ids])
        rank_db = self.rank_db 
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            local_ranks = local_ranks_batch[i]
            relevant_set = ground_truth_sets[i]
            qid = str(qids[i])
            query_prefix_map = rank_db.get(qid, {})

            seq_rewards = [0.0] * len(gen_ids)
            start_idx = 0
            end_idx = len(gen_ids)
            content_ids = gen_ids[start_idx:end_idx]

            # --- 1. 计算全局奖励 ---
            key_str = ",".join(map(str, content_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)

            r_global = 0.0
            if decoded_docid and decoded_docid in relevant_set:
                r_global += 2.0  # Lambda_gt

            # === [修改点] 权重恒为 1.0 ===
            decision_weights = [1.0] * (end_idx - start_idx)
            total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0
            
            # --- 2. 计算每一步奖励 ---
            current_prefix_ids = []
            for t in range(start_idx, end_idx):
                rank = local_ranks[t]
                
                # 获取候选数 (仅用于判断唯一路径给基础分)
                prefix_key = ",".join(map(str, current_prefix_ids))
                if prefix_key != "": prefix_key += "," 
                candidates = query_prefix_map.get(prefix_key, {})
                num_candidates = len(candidates)

                # Stepwise: 正常计算
                step_r = 0.0
                if rank >= 100:
                    step_r = -0.1 
                elif num_candidates == 1:
                    step_r = 0.1 
                else:
                    step_r = 1.0 / math.log(rank + 2)

                # Global: 平均分配
                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight if len(decision_weights) > t else 0
                    seq_rewards[t] = step_r + r_global * dist_weight
                else:
                    seq_rewards[t] = step_r
                
                current_prefix_ids.append(gen_ids[t])

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards
    

    # 消融实验 B：w/o Stepwise Reward
    # 逻辑：强制每一步的局部奖励（Dense Guidance）为 0。模型只能依靠被分配下来的 Global Reward 进行学习。这也验证了“Reward Sparsity”问题。
    def reward_function_only_global(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 2] w/o Stepwise Reward (Only Global GT)
        - 局部奖励 (Stepwise): 强制为 0
        - 全局奖励分配: 正常计算分支权重 (保留拓扑信息)
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        rank_db = self.rank_db 
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            relevant_set = ground_truth_sets[i]
            qid = str(qids[i])
            query_prefix_map = rank_db.get(qid, {})

            seq_rewards = [0.0] * len(gen_ids)
            start_idx = 0
            end_idx = len(gen_ids)
            content_ids = gen_ids[start_idx:end_idx]

            # --- 1. 计算全局奖励 ---
            key_str = ",".join(map(str, content_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)

            r_global = 0.0
            if decoded_docid and decoded_docid in relevant_set:
                r_global += 2.0 

            # --- 2. 计算分支权重 (保留正常逻辑) ---
            decision_weights = []
            if r_global > 0:
                current_prefix_ids = []
                for t in range(start_idx, end_idx):
                    prefix_key = ",".join(map(str, current_prefix_ids))
                    if prefix_key != "": prefix_key += ",1" 
                    candidates = query_prefix_map.get(prefix_key, {})
                    num_c = len(candidates)
                    weight = math.log1p(num_c) # 正常权重
                    decision_weights.append(weight)
                    current_prefix_ids.append(gen_ids[t])
                
                total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # --- 3. 计算奖励 (Stepwise 归零) ---
            for t in range(start_idx, end_idx):
                # === [修改点] Stepwise 强制为 0 ===
                step_r = 0.0 

                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight if len(decision_weights) > t else 0
                    seq_rewards[t] = step_r + r_global / len(gen_ids)
                else:
                    seq_rewards[t] = step_r # 即 0.0

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards




    # 3. 消融实验 C：w/o Dense Guidance (Rank-agnostic)
    # 逻辑：保留 Stepwise 奖励，但不使用 Rank 大小来区分好坏。只要路径有效（即在 Trie 内，Rank < 100），就给一个固定的常数奖励（例如 0.2）。这验证了“Dense Teacher”提供的排名质量的重要性。
    # def reward_function_rank_agnostic(self, prompts, completions, completion_ids, **kwargs):
    #     """
    #     [消融实验 3] w/o Dense Guidance (Rank-agnostic)
    #     - 局部奖励 (Stepwise): 只要路径有效就给固定分 (忽略具体 rank)
    #     - 全局奖励分配: 正常计算分支权重
    #     """
    #     batch_token_rewards = []
    #     ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
    #     local_ranks_batch = kwargs.get("local_ranks", [[1]*len(c) for c in completion_ids])
    #     rank_db = self.rank_db 
    #     qids = kwargs.get("qid", [])

    #     for i in range(len(completion_ids)):
    #         gen_ids = completion_ids[i]
    #         local_ranks = local_ranks_batch[i]
    #         relevant_set = ground_truth_sets[i]
    #         qid = str(qids[i])
    #         query_prefix_map = rank_db.get(qid, {})

    #         seq_rewards = [0.0] * len(gen_ids)
    #         start_idx = 0
    #         end_idx = len(gen_ids)
    #         content_ids = gen_ids[start_idx:end_idx]

    #         # --- 1. 计算全局奖励 ---
    #         key_str = ",".join(map(str, content_ids))
    #         decoded_docid = self.encoded_key_to_original_docid.get(key_str)
    #         r_global = 0.0
    #         if decoded_docid and decoded_docid in relevant_set:
    #             r_global += 5.0

    #         # --- 2. 计算分支权重 (正常) ---
    #         decision_weights = []
    #         if r_global > 0:
    #             current_prefix_ids = []
    #             for t in range(start_idx, end_idx):
    #                 prefix_key = ",".join(map(str, current_prefix_ids))
    #                 if prefix_key != "": prefix_key += ",1" 
    #                 candidates = query_prefix_map.get(prefix_key, {})
    #                 num_c = len(candidates)
    #                 weight = math.log1p(num_c)
    #                 decision_weights.append(weight)
    #                 current_prefix_ids.append(gen_ids[t])
    #             total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

    #         # --- 3. 计算每一步奖励 ---
    #         current_prefix_ids = []
    #         for t in range(start_idx, end_idx):
    #             rank = local_ranks[t]
                
    #             # Stepwise: 忽略具体 rank 大小
    #             step_r = 0.0
    #             if rank >= 100:
    #                 step_r = -0.1 # 路径无效，依然惩罚
    #             else:
    #                 # === [修改点] 只要有效，给固定分，不再用 1/log(rank) ===
    #                 step_r = 0.2  # 固定奖励常数

    #             # Global: 正常分配
    #             if r_global > 0:
    #                 dist_weight = decision_weights[t] / total_weight if len(decision_weights) > t else 0
    #                 seq_rewards[t] = step_r + r_global * dist_weight
    #             else:
    #                 seq_rewards[t] = step_r
                
    #             current_prefix_ids.append(gen_ids[t])

    #         batch_token_rewards.append(seq_rewards)
        
    #     return batch_token_rewards

    def reward_function_rank_agnostic(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3] w/o Dense Guidance (Rank-agnostic)
        - 局部奖励 (Stepwise): 只要路径有效就给固定分 (忽略具体 rank)
        - 全局奖励分配: 正常计算分支权重
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            query_prefix_map = self.rank_db.get(qid, {})

            # --- [关键修改：模块化调用] ---
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            # ---------------------------

            seq_rewards = [0.0] * len(gen_ids)
            
            # --- 1. 计算全局奖励 ---
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            r_global = 2.0 if (decoded_docid and decoded_docid in relevant_set) else 0.0


            if decoded_docid is not None:    # 正确解码的奖励
                r_global += 1
            # --- 2. 计算决策权重 (Branching Weight) ---
            # 依然需要遍历一次来获取拓扑权重
            decision_weights = []
            curr_prefix = []
            for t in range(len(gen_ids)):
                p_key = ",".join(map(str, curr_prefix))
                
                if p_key != "": p_key += ",1" 
                num_c = len(query_prefix_map.get(p_key, {}))
                decision_weights.append(math.log1p(num_c))
                curr_prefix.append(gen_ids[t].item() if hasattr(gen_ids[t], 'item') else gen_ids[t])
            
            total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # --- 3. 计算最终 Token Reward ---
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # Stepwise: Rank-agnostic 逻辑，有效就给 0.2
                step_r = 0.2 if rank < 100 else -0.1
                
                # Global 权重分配_compute_loss
                dist_weight = decision_weights[t] / total_weight
                
                if r_global > 0:
                    seq_rewards[t] = step_r + r_global * dist_weight
                else:
                    seq_rewards[t] = step_r

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards


# 4. 优化版实验：针对 R@1 优化的混合奖励函数
    # 逻辑：结合 Rank 敏感的局部奖励 + 时间衰减的全局分配 + Top-K 软标签
    def reward_function_optimized(self, prompts, completions, completion_ids, **kwargs):
        """
        [New] Optimized Reward Function for High R@1
        Features:
        1. Rank-Sensitive Local Reward: 强区分 Rank 1 vs Rank 10
        2. Time-Decay: 全局奖励更多分配给头部 Token
        3. Soft-Label: 命中 Top-10 但非 GT 时给部分分
        """
        batch_token_rewards = []
        
        # 获取输入数据
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        top_100_docids_batch = kwargs.get("top_100_docids", [[]] * len(completion_ids))
        local_ranks_batch = kwargs.get("local_ranks", [[1]*len(c) for c in completion_ids])
        rank_db = self.rank_db 
        qids = kwargs.get("qid", [])

        # 时间衰减因子 (建议 0.9 或 0.95)
        gamma = 0.9
        # gamma = getattr(self, 'gamma', 0.9) 

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            local_ranks = local_ranks_batch[i]
            relevant_set = ground_truth_sets[i]
            top_100_list = top_100_docids_batch[i]
            qid = str(qids[i])
            query_prefix_map = rank_db.get(qid, {})

            seq_rewards = [0.0] * len(gen_ids)
            start_idx = 0
            end_idx = len(gen_ids)
            content_ids = gen_ids[start_idx:end_idx]

            # --- 1. 计算全局奖励 (Global Reward) ---
            key_str = ",".join(map(str, content_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)

            r_global = 0.0
            
            # Case A: 命中 Ground Truth (最高奖励)
            if decoded_docid and decoded_docid in relevant_set:
                r_global = 2.0
            
            # Case B: [New] 命中 Dense Top-10 (软标签)
            # 如果没命中 GT，但在这个 Query 的 Top 100 召回列表的前 10 名里，也给一点分
            elif decoded_docid and decoded_docid in top_100_list:
                try:
                    rank_in_dense = top_100_list.index(decoded_docid)
                    if rank_in_dense < 10:  # 仅奖励 Top 10
                        # 奖励衰减：Rank 0 -> 0.5, Rank 9 -> 0.05
                        r_global = 0.5 * (1.0 - rank_in_dense / 10.0)
                except ValueError:
                    pass

            # --- 2. 计算决策权重 (Decision Weights) ---
            # 引入时间衰减 gamma^t，让模型更关注早期的 token
            decision_weights = []
            
            if r_global > 0:
                current_prefix_ids = []
                for t in range(start_idx, end_idx):
                    prefix_key = ",".join(map(str, current_prefix_ids))
                    if prefix_key != "": prefix_key += ",1"
                    
                    candidates = query_prefix_map.get(prefix_key, {})
                    num_c = len(candidates)
                    
                    # [New] Weight = 拓扑分支难度 * 时间衰减
                    # 越靠前的步骤，(gamma ** t) 越大，分配到的 Global Reward 越多
                    topo_weight = math.log1p(num_c)
                    time_weight = gamma ** (t - start_idx) 
                    
                    weight = topo_weight * time_weight
                    decision_weights.append(weight)
                    
                    current_prefix_ids.append(gen_ids[t])
                
                total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # --- 3. 计算每一步的奖励 (Stepwise Reward) ---
            current_prefix_ids = []
            for t in range(start_idx, end_idx):
                rank = local_ranks[t] # 当前前缀在 Trie 中能导向的最优 rank
                
                prefix_key = ",".join(map(str, current_prefix_ids))
                if prefix_key != "": prefix_key += ","
                candidates = query_prefix_map.get(prefix_key, {})
                num_candidates = len(candidates)

                step_r = 0.0
                
                # 惩罚无效路径
                if rank >= 100:
                    step_r = -0.1 
                # 奖励有效路径
                else:
                    if num_candidates == 1:
                        step_r = 0.1 # 唯一路径给基础分
                    else:
                        # [New] Rank-Sensitive Reward
                        # Rank 1: 0.5
                        # Rank 10: 0.45
                        # Rank 50: 0.25
                        # Rank 100: ~0.0
                        # 公式：0.5 * (1 - normalized_rank)
                        normalized_rank = max(0, rank - 1) / 100.0
                        step_r = 0.5 * (1.0 - normalized_rank)

                # 叠加 Global Reward
                if r_global > 0:
                    # 分配逻辑
                    dist_weight = decision_weights[t] / total_weight if len(decision_weights) > t else 0
                    seq_rewards[t] = step_r + r_global * dist_weight
                else:
                    seq_rewards[t] = step_r
                
                current_prefix_ids.append(gen_ids[t])

            batch_token_rewards.append(seq_rewards)
            
        return batch_token_rewards


    def reward_function_try_10(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3] w/o Dense Guidance (Rank-agnostic)
        - 局部奖励 (Stepwise): 只要路径有效就给固定分 (忽略具体 rank)
        - 全局奖励分配: 正常计算分支权重
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]

            # --- [关键修改：模块化调用] ---
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            # ---------------------------

            seq_rewards = [0.0] * len(gen_ids)
            
            # --- 1. 计算全局奖励 ---
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            r_global = 2.0 if (decoded_docid and decoded_docid in relevant_set) else 0.0

            if decoded_docid is not None:    # 正确解码的奖励
                r_global += 1

            # --- 3. 计算最终 Token Reward ---
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # Stepwise: Rank-agnostic 逻辑，有效就给 0.2
                step_r = 1/math.log1p(rank) if rank <= 100 else -1
                
                # Global 权重分配
                if r_global > 0:
                    seq_rewards[t] = step_r + r_global/len(gen_ids)
                else:
                    seq_rewards[t] = step_r

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards



    def reward_function_try_0(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3] w/o Dense Guidance (Rank-agnostic)
        - 局部奖励 (Stepwise): 只要路径有效就给固定分 (忽略具体 rank)
        - 全局奖励分配: 正常计算分支权重
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            query_prefix_map = self.rank_db.get(qid, {})

            # --- [关键修改：模块化调用] ---
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            # ---------------------------

            seq_rewards = [0.0] * len(gen_ids)
            
            # --- 1. 计算全局奖励 ---
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            r_global = 1.0 if (decoded_docid and decoded_docid in relevant_set) else 0.0

            # --- 3. 计算最终 Token Reward ---
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # Stepwise: Rank-agnostic 逻辑，有效就给 0.2
                step_r = 0.2 if rank <= 100 else -0.1
                
                # Global 权重分配
                # dist_weight = decision_weights[t] / total_weight
                
                if r_global > 0:
                    seq_rewards[t] = step_r + r_global 
                else:
                    seq_rewards[t] = step_r
                
                # seq_rewards[t] += r_real_doc   # 额外加上解码到真实文档的奖励  
                ## DEBUG
                # noise = random.uniform(-2, 2)
                # seq_rewards[t] += noise 

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards
    
    
    def reward_function_v2(self, prompts, completions, completion_ids, **kwargs):
        """
        改进版奖励函数：
        1. 消除长度陷阱：有效路径不给分，只在命中时给终点大分。
        2. 强力约束：偏离 Trie 树给予较大负分。
        3. 简洁奖励：微小的长度惩罚。
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            
            # 获取每个 token 对应的路径排名
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            seq_len = len(gen_ids)
            seq_rewards = [0.0] * seq_len
            
            # --- 1. 计算全局命中奖励 (仅一次性判断) ---
            # 只有完整生成的 docid 命中 GT 才给分
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            
            # 基础全局分值设定为 10.0，不分摊
            hit_reward = 10.0 if (decoded_docid and decoded_docid in relevant_set) else 0.0

            # --- 2. Token-Level 逻辑分配 ---
            is_off_track = False
            for t in range(seq_len):
                # A. 长度惩罚 (非常微小，促使模型尽早输出 EOS)
                step_r = -0.01 
                
                rank = local_ranks[t]
                
                # B. 路径合法性判断
                if rank >= 100:
                    is_off_track = True
                    step_r = -1.0  # 偏离路径给重罚
                elif is_off_track:
                    step_r = -1.0  # 一旦偏离，后面步步重罚
                
                # C. 命中奖励：只加在最后一个有效 Token 上 (脉冲式)
                # 这样可以给模型一个强烈的信号：这一串路径最终导向了正确答案
                if t == (seq_len - 1) and hit_reward > 0 and not is_off_track:
                    seq_rewards[t] = step_r + hit_reward
                else:
                    seq_rewards[t] = step_r

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards




    def reward_function_generative_retrieval(self, prompts, completions, completion_ids, **kwargs):
            """
            专为生成式检索 (GR) 设计的 Token-level 奖励函数
            结合了：密集前缀引导 + 节点难度加权 + 全局命中奖励 + 同组去重惩罚
            """
            batch_token_rewards = []
            
            # 获取每条 Query 对应的相关文档集合 (Ground Truth)
            # 假设 relevant_docid_set 里面存的是 DocID 字符串 (如 "doc_1234")
            ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
            qids = kwargs.get("qid", [])

            # 遍历当前 Batch 中的每一条生成结果
            for i in range(len(completion_ids)):
                gen_ids = completion_ids[i]
                qid = str(qids[i])
                relevant_set = ground_truth_sets[i]
                
                # 初始化该条序列每个 Token 的奖励为 0
                seq_rewards = [0.0] * len(gen_ids)
                
                # 将相关文档集合的字符串，提前转换为 Token ID 列表的形式，方便做前缀匹配
                # 假设你有一个方法可以做到： "doc_123" -> [10, 25, 36, 99]
                # 如果你有多个相关的 ground truth，就转化出多个 list
                gt_token_lists = self._get_gt_token_lists_for_query(relevant_set)

                # -----------------------------------------------------------------
                # 状态追踪标志
                # -----------------------------------------------------------------
                is_on_right_track = True  # 标记当前是否还在正确的路径上
                
                # --- 开始逐个 Token 评估 ---
                for t in range(len(gen_ids)):
                    current_token = gen_ids[t].item() if hasattr(gen_ids[t], 'item') else gen_ids[t]
                    current_prefix = gen_ids[:t+1]
                    
                    # 1. 前缀匹配检测 (判断当前走到 t 步时，是不是任何一个 GT 的前缀)
                    match_any_gt = any(
                        self._is_prefix(current_prefix, gt_seq) 
                        for gt_seq in gt_token_lists
                    )

                    if match_any_gt and is_on_right_track:
                        # ==========================================
                        # 【场景 A】：走在正确的康庄大道上
                        # ==========================================
                        # 基础过路费奖励
                        step_r = 0.5 
                        
                        # 难度加权：通过 Trie 树或 rank_db 查一下当前节点有几个合法子节点
                        # 子节点越多，选对的含金量越高
                        p_key = ",".join(map(str, gen_ids[:t]))
                        if p_key != "": p_key += ",1" 
                        num_children = len(self.rank_db.get(p_key, {})) # 或者查 docid_trie
                        
                        difficulty_weight = math.log1p(num_children) * 0.1 # 缩放因子 0.1 可调
                        
                        seq_rewards[t] = step_r + difficulty_weight

                    else:
                        # ==========================================
                        # 【场景 B】：一旦走错一步，后续全错
                        # ==========================================
                        is_on_right_track = False # 永久打断
                        
                        # 走错了，给一个温和的惩罚。
                        # 不要给太大的负数（比如 -10），否则模型会害怕探索，甚至学会提前截断
                        seq_rewards[t] = -0.2

                # ==========================================
                # 【场景 C】：全局奖励 (到达终点且完全正确)
                # ==========================================
                # 将生成的完整序列转回字符串 DocID
                key_str = ",".join(map(str, gen_ids))
                decoded_docid = self.encoded_key_to_original_docid.get(key_str, None)
                
                # 只有最后一步才结算全局奖励 (加在序列末尾的 Token 上)
                if decoded_docid and decoded_docid in relevant_set:
                    # 命中目标！给予超级大奖
                    seq_rewards[-1] += 15.0 
                else:
                    # 完整生成完发现不对，额外给个终点惩罚
                    seq_rewards[-1] -= 2.0

                batch_token_rewards.append(seq_rewards)
                
            return batch_token_rewards

    # 辅助函数：判断 list1 是否是 list2 的前缀
    def _is_prefix(self, prefix, full_seq):
        if len(prefix) > len(full_seq):
            return False
        # 逐元素比较
        for p, f in zip(prefix, full_seq):
            if p != f:
                return False
        return True
    
    # 辅助函数：把 Ground truth 的 docid 字符串转成 Token ID 序列
    def _get_gt_token_lists_for_query(self, relevant_set):
        gt_lists = []
        for docid_str in relevant_set:
            # 根据你的字典转一下，例如 "MSMARCO_123" -> [45, 88, 99]
            encoded_list = self.original_to_encoded_list.get(docid_str, [])
            if encoded_list:
                gt_lists.append(encoded_list)
        return gt_lists



    def reward_function_generative_retrieval_1(self, prompts, completions, completion_ids, **kwargs):
        batch_token_rewards = []
        
        # 提取数据
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        top_100_docids_batch = kwargs.get("top_100_docids", [[]] * len(completion_ids)) 
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            top_100_list = top_100_docids_batch[i] if i < len(top_100_docids_batch) else []
            query_prefix_map = self.rank_db.get(qid, {})
            
            seq_rewards = [0.0] * len(gen_ids)
            
            # --- 1. 获取 Stepwise 排名 (Local Ranks) ---
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            
            # --- 2. 评估全局终点 (Global Hit) ---
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            
            r_global = 0.0
            if decoded_docid:
                if decoded_docid in relevant_set:
                    # GT 命中超级大奖 (总资金池)
                    r_global = 15.0
                elif decoded_docid in top_100_list:
                    # 软标签命中 (Dense Top 100)
                    rank_idx = top_100_list.index(decoded_docid)
                    # 排名越高分越多 (0.5 ~ 2.0)
                    r_global = 2.0 * (1.0 - rank_idx / 100.0)


            # --- 3. 计算拓扑分支难度 (Decision Weights) ---
            # 这决定了全局奖金池怎么分给每一个 Token
            decision_weights = []
            curr_prefix = []
            for t in range(len(gen_ids)):
                p_key = ",".join(map(str, curr_prefix))
                if p_key != "": 
                    p_key += ",1" 
                
                num_c = len(query_prefix_map.get(p_key, {}))
                decision_weights.append(math.log1p(num_c))
                
                token_val = gen_ids[t].item() if hasattr(gen_ids[t], 'item') else gen_ids[t]
                curr_prefix.append(token_val)
            
            total_weight = sum(decision_weights)
            # 防止全 0 除法
            total_weight = total_weight if total_weight > 0 else 1.0

            # --- 4. 融合与分配 (Stepwise + Distributed Global) ---
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # A. 基础过路费 (Stepwise 密集引导)
                # 不用负数倒扣，用正数引导，防止 KL 散度爆炸和遗忘
                if rank <= 100:
                    step_r = 1.0 / math.log1p(rank)  # 排名越前，分数越高 (约 0.2 ~ 1.4)
                else:
                    step_r = -0.05  # 偏离路径给微弱惩罚
                
                # B. 分配全局奖金
                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight
                    token_global_r = r_global * dist_weight
                else:
                    token_global_r = 0.0
                
                # C. 当前 Token 最终得分
                seq_rewards[t] = step_r + token_global_r

            batch_token_rewards.append(seq_rewards)
            
        return batch_token_rewards




    def reward_function_noly_step(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3] w/o Dense Guidance (Rank-agnostic)
        - 奖励是1/log(rank + 1)，rank <= 100 才有奖励，且不分配全局奖励
        - rank > 100 奖励为0
        """
        batch_token_rewards = []
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])

            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            seq_rewards = [0.0] * len(gen_ids)
            
            for t in range(len(gen_ids)):
                rank = local_ranks[t]            
                if rank <= 100:
                    seq_rewards[t] = 1/math.log1p(rank)

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards
    

    def reward_function_decay_state(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3 增强版] w/o Dense Guidance + Prefix Decay
        - 基础奖励: 1/log1p(rank)
        - 前缀折扣: 一旦出现高 Rank (> 20)，后续所有奖励进入衰减状态
        """
        batch_token_rewards = []
        qids = kwargs.get("qid", [])
        
        # 调优参数：你可以根据实验调整
        RANK_THRESHOLD = 20    # 触发衰减的排名阈值
        DECAY_RATE = 0.9       # 每次触发后的折扣率（越小惩罚越狠）

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            
            # 假设这是你获取每一步 rank 的方法
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            
            seq_rewards = [0.0] * len(gen_ids)
            current_multiplier = 1.0  # 初始倍率为 100%
            
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # --- 前缀折扣逻辑 ---
                # 如果当前 token 排名太靠后，降低后续所有 token 的“信用分”
                if rank > RANK_THRESHOLD:
                    current_multiplier *= DECAY_RATE
                
                # --- 基础奖励计算 ---
                if rank <= 100:
                    # 原始奖励 * 累积的折扣倍率
                    base_reward = 1.0 / math.log1p(rank)
                    seq_rewards[t] = base_reward * current_multiplier
                else:
                    seq_rewards[t] = 0.0

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards

    def reward_function_decay_state_all(self, prompts, completions, completion_ids, **kwargs):
        """
        [终极融合版] 分支权重分配 + 前缀衰减 + 全局防作弊
        解决问题：
        1. 解决模型提前输出 EOS 骗分的作弊行为 (长度崩溃问题)。
        2. 解决长序列前面 Token 拿不到全局反馈的问题 (TRL 无 Return-to-go 盲区)。
        """
        import math
        batch_token_rewards = []
        
        qids = kwargs.get("qid", [])
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        
        # --- 调优参数 ---
        RANK_THRESHOLD = 20    # 触发衰减的排名阈值
        DECAY_RATE = 0.9       # 偏离后的惩罚衰减率
        GLOBAL_HIT_REWARD = 15.0     # 命中 Ground Truth 的超级大奖
        GLOBAL_VALID_REWARD = 0.5    # 没命中，但生成了一个完整的合法 DocID (安慰奖)
        GLOBAL_HACK_PENALTY = -2.0   # 作弊惩罚：提前截断、乱码、不完整的 DocID

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            query_prefix_map = self.rank_db.get(qid, {})
            
            # 获取每一步 rank
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            seq_rewards = [0.0] * len(gen_ids)
            
            # ==========================================
            # 1. 终局裁判 (Global Check)
            # ==========================================
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            
            r_global = 0.0
            if decoded_docid:
                # 能够成功解码，说明模型跑完了完整的 Trie 树路径，没有作弊截断
                if decoded_docid in relevant_set:
                    r_global = GLOBAL_HIT_REWARD  # 彻底找对！
                else:
                    r_global = GLOBAL_VALID_REWARD # 虽然找错了，但格式和生成过程很乖，给小奖
            else:
                # 致命作弊：根本没生成合法的 DocID 就提前结束了（引发你图表中长度跌到 2 的元凶）
                r_global = GLOBAL_HACK_PENALTY
                

            # ==========================================
            # 2. 拓扑难度计算 (用于分配正向全局奖励)
            # ==========================================
            decision_weights = []
            curr_prefix = []
            for t in range(len(gen_ids)):
                p_key = ",".join(map(str, curr_prefix))
                if p_key != "": 
                    p_key += ",1" 
                
                num_c = len(query_prefix_map.get(p_key, {}))
                decision_weights.append(math.log1p(num_c))
                
                token_val = gen_ids[t].item() if hasattr(gen_ids[t], 'item') else gen_ids[t]
                curr_prefix.append(token_val)
                
            total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # ==========================================
            # 3. 逐 Token 结算 (Stepwise + Decay + 分配)
            # ==========================================
            current_multiplier = 1.0  # 初始信用倍率
            
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # --- A. 前缀折扣逻辑 ---
                if rank > RANK_THRESHOLD:
                    current_multiplier *= DECAY_RATE
                
                # --- B. 局部引导分数 ---
                if rank <= 100:
                    step_r = (1.0 / math.log1p(rank)) * current_multiplier
                else:
                    step_r = -0.1 # 绝对偏离的微弱惩罚
                
                # --- C. 全局大奖分配 ---
                # 如果是正向大奖，科学地切碎分给每一个关键 Token
                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight
                    token_global_r = r_global * dist_weight
                else:
                    token_global_r = 0.0
                    
                # 汇总当前 Token 分数
                seq_rewards[t] = step_r + token_global_r
                
                # --- D. 绝杀作弊者 ---
                # 如果是负向惩罚（作弊截断），直接狠狠砸在最后一个 Token 上！
                # 这会立刻让模型意识到：提前结束是死路一条！
                if t == len(gen_ids) - 1 and r_global < 0:
                    seq_rewards[t] += r_global

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards
    def reward_function_decay_state_all_without_GLOBAL_VALID_REWARD(self, prompts, completions, completion_ids, **kwargs):
        """
        [终极融合版] 分支权重分配 + 前缀衰减 + 全局防作弊
        解决问题：
        1. 解决模型提前输出 EOS 骗分的作弊行为 (长度崩溃问题)。
        2. 解决长序列前面 Token 拿不到全局反馈的问题 (TRL 无 Return-to-go 盲区)。
        """
        import math
        batch_token_rewards = []
        
        qids = kwargs.get("qid", [])
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        
        # --- 调优参数 ---
        RANK_THRESHOLD = 20    # 触发衰减的排名阈值
        DECAY_RATE = 0.9       # 偏离后的惩罚衰减率
        GLOBAL_HIT_REWARD = 15.0     # 命中 Ground Truth 的超级大奖

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            query_prefix_map = self.rank_db.get(qid, {})
            
            # 获取每一步 rank
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            seq_rewards = [0.0] * len(gen_ids)
            
            # ==========================================
            # 1. 终局裁判 (Global Check)
            # ==========================================
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            
            r_global = 0.0
            if decoded_docid:
                # 能够成功解码，说明模型跑完了完整的 Trie 树路径，没有作弊截断
                if decoded_docid in relevant_set:
                    r_global = GLOBAL_HIT_REWARD  # 彻底找对！


            # ==========================================
            # 2. 拓扑难度计算 (用于分配正向全局奖励)
            # ==========================================
            decision_weights = []
            curr_prefix = []
            for t in range(len(gen_ids)):
                p_key = ",".join(map(str, curr_prefix))
                if p_key != "": 
                    p_key += ",1" 
                
                num_c = len(query_prefix_map.get(p_key, {}))
                decision_weights.append(math.log1p(num_c))
                
                token_val = gen_ids[t].item() if hasattr(gen_ids[t], 'item') else gen_ids[t]
                curr_prefix.append(token_val)
                
            total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # ==========================================
            # 3. 逐 Token 结算 (Stepwise + Decay + 分配)
            # ==========================================
            current_multiplier = 1.0  # 初始信用倍率
            
            for t in range(len(gen_ids)):
                rank = local_ranks[t]
                
                # --- A. 前缀折扣逻辑 ---
                if rank > RANK_THRESHOLD:
                    current_multiplier *= DECAY_RATE
                
                # --- B. 局部引导分数 ---
                if rank <= 100:
                    step_r = (1.0 / math.log1p(rank)) * current_multiplier
                else:
                    step_r = -0.1 # 绝对偏离的微弱惩罚
                
                # --- C. 全局大奖分配 ---
                # 如果是正向大奖，科学地切碎分给每一个关键 Token
                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight
                    token_global_r = r_global * dist_weight
                else:
                    token_global_r = 0.0
                    
                # 汇总当前 Token 分数
                seq_rewards[t] = step_r + token_global_r
                


            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards
    


    def reward_function_pulsed(self, prompts, completions, completion_ids, **kwargs):
        """
        [优化版] 脉冲式奖励函数
        1. 消除长度陷阱：路径中间不给正分。
        2. 严格路径约束：一旦 Rank > 20 立即给惩罚并截断后续奖励。
        3. 全局命中大奖：只有在生成完整且合法的 DocID 时才发放。
        """
        batch_token_rewards = []
        ground_truth_sets = kwargs.get("relevant_docid_set", [set()] * len(completion_ids))
        qids = kwargs.get("qid", [])

        for i in range(len(completion_ids)):
            gen_ids = completion_ids[i]
            qid = str(qids[i])
            relevant_set = ground_truth_sets[i]
            
            # 获取每一步的局部排名
            local_ranks = self._get_local_ranks_for_sequence(qid, gen_ids)
            seq_len = len(gen_ids)
            seq_rewards = [0.0] * seq_len
            
            # --- A. 终局判定 (Global Check) ---
            key_str = ",".join(map(str, gen_ids))
            decoded_docid = self.encoded_key_to_original_docid.get(key_str)
            
            # 判定是否彻底命中 GT
            is_hit = (decoded_docid and decoded_docid in relevant_set)
            
            # --- B. 逐 Token 奖励分配 ---
            off_track_t = -1  # 记录第一次偏离的位置
            for t in range(seq_len):
                rank = local_ranks[t]
                
                # 1. 路径惩罚：一旦排名靠后（比如不在前20），给予阶梯式扣分
                if rank > 20:
                    if off_track_t == -1: off_track_t = t
                    seq_rewards[t] = -0.5  # 偏离惩罚
                elif rank > 1:
                    seq_rewards[t] = -0.05 # 走在边缘的微弱惩罚，促使模型追求 Rank 1
                else:
                    seq_rewards[t] = 0.0   # 走在 Rank 1 路径上不加分也不扣分
                
                # 2. 长度陷阱防御：如果序列太短且没解码出东西，给一个结尾重罚
                if t == seq_len - 1 and not decoded_docid:
                    seq_rewards[t] -= 2.0

            # --- C. 脉冲奖励 (The Impulse) ---
            # 只有在没有偏离路径且最终命中的情况下，在最后一个 Token 加上巨额奖励
            if is_hit and off_track_t == -1:
                # 这里的 20.0 是为了盖过所有的微弱惩罚，让命中成为唯一目标
                seq_rewards[-1] += 20.0 
            
            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards


