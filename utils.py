import os
import sys
import gzip
from tqdm.auto import tqdm
import torch
from collections import defaultdict
from typing import Optional, Union, Dict, Any, List, Set
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    PreTrainedModel,
)
from datasets import Dataset

from trie import Trie

def convert_token_ids_to_key(token_ids: list[int]) -> str:
    
    """
    【核心函数】将 token ID 列表转换为标准化的字符串 key。
    例如：[0, 149, 33, 1, 99] -> "149,33"
    """
    if not token_ids:
        return ""
        
    clean_ids = []
    # 从第一个非 <pad> (0) 的 token 开始
    start_idx = 0
    if token_ids[0] == 0: # T5 decoder anways starts with <pad>
        start_idx = 1

    for token in token_ids[start_idx:]:
        if token == 1:  # 遇到 <eos> (1) 就停止
            break
        clean_ids.append(str(token))
    
    clean_ids.append(str(1)) # 结尾加上 <eos>
    # 用逗号连接，不带任何空格
    return ",".join(clean_ids)


def load_encoded_docids_and_create_map(path):
    """
    加载 docid.txt 文件，并创建两个映射：
    1. 编码字符串 -> 原始ID (用于奖励计算)
    2. 原始ID -> 编码列表 (用于数据预处理)
    同时返回所有编码序列用于构建Trie树。
    """
    encoded_key_to_original = {}
    original_to_encoded_list = {}
    all_encoded_sequences = []

    with open(path, "r") as f:
        for line in tqdm(f, desc="Loading encoded docIDs"):
            # 文件格式是 "[d108472]\t594,858,..."
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            # 1. 清理原始 docid
            original_docid = parts[0].strip("[]").upper() # -> "D108472"
            
            # 2. 解析编码
            pq_ids_str = parts[1]
            encoded_list = [int(x) for x in pq_ids_str.split(",")] # -> [594, 858, ...]
            
            # 3. 创建 key-value 对
            # key 直接是文件中的编码字符串，这是我们的黄金标准
            encoded_key_to_original[pq_ids_str] = original_docid
            original_to_encoded_list[original_docid] = encoded_list
            all_encoded_sequences.append(encoded_list)

    return encoded_key_to_original, original_to_encoded_list, all_encoded_sequences


def build_partial_trie(sequences_chunk):
    # from trie_cpp import Trie
    trie = Trie(sequences_chunk)
    return trie


def load_generative_retrieval_model(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path, use_fast=True)
    base_model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path,
                                                            torch_dtype=torch.float32)

    return base_model, tokenizer

# def load_generative_retrieval_model(args):
#     tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path, use_fast=True)
    
#     # 显式指定 device_map=None
#     base_model = T5ForConditionalGeneration.from_pretrained(
#         args.checkpoint_path,
#         torch_dtype=torch.float32,
#         device_map=None, 
#         low_cpu_mem_usage=False,
#     )

#     # 2. 【新增】深度清洗：清除 model.config 中的残留
#     # 有时候 accelerate 会把 map 写进 config，导致复制时复活
#     if hasattr(base_model.config, "hf_device_map"):
#         del base_model.config.hf_device_map
#     if hasattr(base_model.config, "device_map"):
#         del base_model.config.device_map

#     # 3. 清除模型本身的残留
#     if hasattr(base_model, "hf_device_map"):
#         del base_model.hf_device_map
#     if hasattr(base_model, "device_map"):
#         del base_model.device_map
        
#     # 4. 确保 T5 不会被误认为是并行模型
#     base_model.is_parallelizable = False
#     base_model.model_parallel = False

#     return base_model, tokenizer
# --------------------------------------------------------------------------------
# 步骤 2: 自定义 GRPOTrainer
# --------------------------------------------------------------------------------

def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


def load_query_texts(path):
    mapping = {}
    with gzip.open(path, "rt", encoding='utf-8') as f:
        for line in f:
            topic_id, query = line.strip().split("\t")
            mapping[topic_id] = query
    return mapping


def load_encoded_docids(path):
    # original_to_encoded, encoded_tuple_to_original, all_encoded_sequences = {}, {}, []
    # 我们不再需要 tuple 字典了
    encoded_key_to_original = {}
    all_encoded_sequences = []

    with open(path, "r") as f:
        for line in tqdm(f, desc="Loading encoded docIDs"):
            docid, pq_ids_str = line.strip().split("\t")
            original_docid = docid.strip("[]").upper()
            
            # 从文件字符串创建 token ID 列表
            encoded_list = [int(x) for x in pq_ids_str.split(",")]
            
            # 【关键修改】使用标准函数生成 key
            key = convert_token_ids_to_key([0] + encoded_list + [1]) # 模拟模型生成的格式 <pad>...<eos>
            
            encoded_key_to_original[key] = original_docid
            all_encoded_sequences.append(encoded_list)

    # 返回新的字典
    return encoded_key_to_original, all_encoded_sequences


def load_rankings_and_qrels(path):
    data = {}
    with open(path, "r", encoding='utf-8') as f:
        # 读取并跳过第一行（表头）
        next(f) 
        for line in tqdm(f, desc=f"Loading rankings from {os.path.basename(path)}"):
            parts = line.strip().split('\t')
            if len(parts) != 3: 
                continue
            qid, relevant_docs_str, ranked_docs_str = parts
            relevant_set = set(relevant_docs_str.split(','))
            ranked_list = ranked_docs_str.split(',')
            ranking_map = {docid: i for i, docid in enumerate(ranked_list)}
            data[qid] = {"relevant_docid_set": relevant_set, "dense_ranking_map": ranking_map}
    return data


def create_dataset_with_ranking_list(queries_path, rankings_path, max_samples=None, qrels_map: Optional[Dict[str, Set[str]]] = None):
    queries_dict = load_query_texts(queries_path)
    data_list = []
    with open(rankings_path, "r", encoding='utf-8') as f:
        try:
            next(f) 
        except StopIteration:
            return Dataset.from_list([])
        for line in tqdm(f, desc=f"Processing {os.path.basename(rankings_path)}"):
            parts = line.strip().split('\t')
            if len(parts) != 3: continue
            qid, target_docs_original_id, ranked_docs_original_ids = parts
            prompt = queries_dict.get(qid)
            if prompt:
                item = {
                    "prompt": prompt, 
                    "qid": qid,
                }
                # 如果是为训练集创建数据 (qrels_map is None)，则保留原有字段用于奖励计算
                if qrels_map is None:
                    item["relevant_docid_set"] = list(set(target_docs_original_id.split(','))) if target_docs_original_id else []
                    item["top_100_docids"] = ranked_docs_original_ids.split(',')[:100]
                # 如果是为评估集创建数据，则添加真实标签字段
                else:
                    item["ground_truth_original_docids"] = list(qrels_map.get(qid, set()))
                
                data_list.append(item)

            if max_samples is not None and len(data_list) >= max_samples:
                break
    if not data_list:
        raise ValueError("No valid samples were created.")
    return Dataset.from_list(data_list)


def load_qrels(path: str) -> Dict[str, Set[str]]:
    qrels = defaultdict(set)
    with gzip.open(path, "rt", encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Qrels"):
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, _ = parts
                # 将docid转换为大写以保持一致
                qrels[qid].add(docid.upper())
    return qrels


def docid2string_msmarco(ids: List[int]) -> str:
    """
    Script B 的核心逻辑：去除 0，保留第一个 1，截断后续。
    """
    seq: List[int] = []
    for x in ids:
        if x == 0: # 过滤 BOS/PAD
            continue
        if x == 1: # 遇到 EOS
            seq.append(1)
            break
        seq.append(x)
    return ",".join(map(str, seq))

def safe_lookup(
    key: str,
    table: Dict[str, Union[str, List[str]]], # 兼容 value 是 str 或 list[str]
    fallback: bool = False,
    alt_key: str = None,
) -> List[str]:
    # 辅助函数：统一返回 list
    def _to_list(val):
        if isinstance(val, list): 
            return val
        return [val]

    if key in table:
        return _to_list(table[key])
    
    if fallback and alt_key is not None and alt_key in table:
        return _to_list(table[alt_key])
    
    return [] # 没找到返回空列表