import os
import tqdm
import shelve
from collections import defaultdict
import argparse

# 假设 utils 在同级目录下
from utils import load_encoded_docids_and_create_map, convert_token_ids_to_key, create_dataset_with_ranking_list

def preprocess_rank_map(args):
    # 检查输出目录是否存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 1. 加载 DocID -> Token IDs 的映射
    print(f"Loading DocID Map from {args.encoded_docid_path}...")
    _, original_to_encoded_list, _ = load_encoded_docids_and_create_map(args.encoded_docid_path)
    
    # 2. 读取 Query -> Top100 DocIDs
    print(f"Loading Training Rankings from {args.train_rankings_file}...")
    train_data = create_dataset_with_ranking_list(
        args.train_queries_file, 
        args.train_rankings_file, 
        max_samples=None
    )
    
    db_name = "query_prefix_rank_map_db"
    output_path = os.path.join(args.output_dir, db_name)
    print(f"Generating RELATIVE Prefix Rank Map to {output_path}...")
    
    # 使用 shelve 创建一个持久化字典
    # flag='n' 表示总是创建一个新的空数据库
    with shelve.open(output_path, flag='n') as db:
        
        count = 0
        for sample in tqdm.tqdm(train_data, desc="Processing Queries"):
            qid = str(sample['qid'])
            top_100_docids = sample['top_100_docids']
            
            # --- 第一步：收集绝对排名 ---
            # raw_prefix_map: key="prefix", value={token: absolute_doc_rank}
            raw_prefix_map = {} 
            
            for rank_idx, docid in enumerate(top_100_docids):
                current_abs_rank = rank_idx + 1 # 原始绝对排名 (1-100)
                
                doc_tokens = original_to_encoded_list.get(docid)
                if not doc_tokens:
                    continue
                
                current_prefix_tokens = []
                for token in doc_tokens:
                    # 生成 Key
                    if len(current_prefix_tokens) == 0:
                        prefix_key = ""
                    else:
                        prefix_key = convert_token_ids_to_key(current_prefix_tokens)
                    
                    # 记录该前缀下，该 token 对应的最好绝对排名
                    if prefix_key not in raw_prefix_map:
                        raw_prefix_map[prefix_key] = {}
                    
                    if token not in raw_prefix_map[prefix_key]:
                        raw_prefix_map[prefix_key][token] = current_abs_rank
                    else:
                        # 取最小值（即使是不同文档，只要接这个token，就看最好的那个）
                        raw_prefix_map[prefix_key][token] = min(raw_prefix_map[prefix_key][token], current_abs_rank)
                    
                    current_prefix_tokens.append(token)
            
            # --- 第二步：转换为相对排名 (Re-ranking) ---
            # final_prefix_map: key="prefix", value={token: relative_rank_1_to_N}
            final_prefix_map = {}
            
            for prefix_key, candidates in raw_prefix_map.items():
                # candidates 是 {token_id: best_absolute_rank}
                # 我们根据 absolute_rank 对 token 进行排序
                # sorted_items: [(token_a, 5), (token_b, 20), (token_c, 99)]
                sorted_items = sorted(candidates.items(), key=lambda x: x[1])
                
                relative_map = {}
                # 重新分配排名：1, 2, 3 ...
                for new_rank_idx, (token_id, old_abs_rank) in enumerate(sorted_items):
                    # new_rank 从 1 开始
                    relative_map[token_id] = new_rank_idx + 1
                
                final_prefix_map[prefix_key] = relative_map

            # 保存处理后的相对排名 Map
            db[qid] = final_prefix_map
            count += 1

    print(f"Pre-processing Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- 已经在 default 中填好了你的路径 ---
    
    # 1. 编码后的 DocID 映射文件 (用于获取 docid -> token_ids)
    parser.add_argument(
        "--encoded_docid_path", 
        type=str, 
        default="/data1/fengjun/workspace/GR-GRPO/GR-RL/data/encode_docid/url_title_docid.txt"
    )
    
    # 2. 训练集 Query 文件
    parser.add_argument(
        "--train_queries_file", 
        type=str, 
        default="/data1/fengjun/workspace/GR-GRPO/QWen3-embedding-test/data/msmarco-data/msmarco-doctrain-queries.tsv.gz"
    )
    
    # 3. 训练集 Ranking 文件 (Top-100)

    parser.add_argument(
        "--train_rankings_file", 
        type=str, 
        default="/data1/fengjun/workspace/ly/ddro/masmarco-top100/msmarco-4096-top100-train.tsv"
    )
    
    # 4. 输出目录 (生成的 DB 文件存放位置)
    # 我设置为了 datasets/trie 目录，如果不存在会自动创建
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/data1/fengjun/workspace/ly/ddro/grpo-ddro/msmarco-tu-map-localrank-correct-4096"
    )

    args = parser.parse_args()
    
    preprocess_rank_map(args)