
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
        # === 【修正】统一变量名为 epoch_xxx ===
        self.epoch_total_count = 0
        self.epoch_hit_count = 0


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
                self.epoch_total_count += 1
                # Part A: GT
                if decoded_docid and decoded_docid in relevant_set:
                    r_global += 2
                    self.epoch_hit_count += 1  # 命中了就+1

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
            # 返回 List[List[float]]
            # print("="*60)
            # print({"epoch_total_count": self.epoch_total_count, "epoch_hit_count": self.epoch_hit_count, "epoch_hit_rate": self.epoch_hit_count/self.epoch_total_count if self.epoch_total_count>0 else 0.0})
            # print("="*60)
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
                    seq_rewards[t] = step_r + r_global * dist_weight
                else:
                    seq_rewards[t] = step_r # 即 0.0

            batch_token_rewards.append(seq_rewards)
        
        return batch_token_rewards




    # 3. 消融实验 C：w/o Dense Guidance (Rank-agnostic)
    # 逻辑：保留 Stepwise 奖励，但不使用 Rank 大小来区分好坏。只要路径有效（即在 Trie 内，Rank < 100），就给一个固定的常数奖励（例如 0.2）。这验证了“Dense Teacher”提供的排名质量的重要性。
    def reward_function_rank_agnostic(self, prompts, completions, completion_ids, **kwargs):
        """
        [消融实验 3] w/o Dense Guidance (Rank-agnostic)
        - 局部奖励 (Stepwise): 只要路径有效就给固定分 (忽略具体 rank)
        - 全局奖励分配: 正常计算分支权重
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
                r_global += 5.0

            # --- 2. 计算分支权重 (正常) ---
            decision_weights = []
            if r_global > 0:
                current_prefix_ids = []
                for t in range(start_idx, end_idx):
                    prefix_key = ",".join(map(str, current_prefix_ids))
                    if prefix_key != "": prefix_key += ",1" 
                    candidates = query_prefix_map.get(prefix_key, {})
                    num_c = len(candidates)
                    weight = math.log1p(num_c)
                    decision_weights.append(weight)
                    current_prefix_ids.append(gen_ids[t])
                total_weight = sum(decision_weights) if sum(decision_weights) > 0 else 1.0

            # --- 3. 计算每一步奖励 ---
            current_prefix_ids = []
            for t in range(start_idx, end_idx):
                rank = local_ranks[t]
                
                # Stepwise: 忽略具体 rank 大小
                step_r = 0.0
                if rank >= 100:
                    step_r = -0.1 # 路径无效，依然惩罚
                else:
                    # === [修改点] 只要有效，给固定分，不再用 1/log(rank) ===
                    step_r = 0.2  # 固定奖励常数

                # Global: 正常分配
                if r_global > 0:
                    dist_weight = decision_weights[t] / total_weight if len(decision_weights) > t else 0
                    seq_rewards[t] = step_r + r_global * dist_weight
                else:
                    seq_rewards[t] = step_r
                
                current_prefix_ids.append(gen_ids[t])

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
            self.epoch_total_count += 1
            
            # Case A: 命中 Ground Truth (最高奖励)
            if decoded_docid and decoded_docid in relevant_set:
                r_global = 2.0
                self.epoch_hit_count += 1
            
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









