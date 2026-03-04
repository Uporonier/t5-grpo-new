import os
import pickle
from tqdm import tqdm
from utils import load_encoded_docids_and_create_map

def main():
    # 1. 路径配置
    RANKINGS_FILE = "/data1/fengjun/workspace/GR-GRPO/GR-RL/data/ranked_results_with_qrels_and_top100_docs.tsv"
    ENCODED_DOCID_PATH = "/data1/fengjun/workspace/GR-GRPO/GR-RL/data/encode_docid/url_title_docid.txt"
    # 保存为 paths 而不是 trie
    SAVE_PATH = "/data1/fengjun/workspace/ly/ddro/grpo-ddro/trie/query_top100_paths.pkl"
    
    # 2. 加载 原始ID -> Token ID List 的映射
    print("Loading Encoded DocID Map...")
    _, original_to_encoded_list, _ = load_encoded_docids_and_create_map(ENCODED_DOCID_PATH)
    
    # 3. 读取 Rankings 文件并保存路径列表
    query_paths_map = {} # Key: qid, Value: List[List[int]]
    
    print(f"Processing {RANKINGS_FILE}...")
    with open(RANKINGS_FILE, "r", encoding='utf-8') as f:
        header = next(f) 
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            qid = parts[0]
            ranked_docs_str = parts[2]
            
            if not ranked_docs_str:
                continue
                
            top_100_docids = ranked_docs_str.split(',')[:100]
            
            sequences = []
            for docid in top_100_docids:
                if docid in original_to_encoded_list:
                    # 补上 <pad> (0) 和 <eos> (1) 以匹配 T5 模型行为
                    # 如果 original_to_encoded_list 里已经包含了 EOS，请根据实际情况调整
                    seq = [0] + original_to_encoded_list[docid] + [1]
                    sequences.append(seq)
            
            if sequences:
                query_paths_map[qid] = sequences
    
    # 4. 保存
    print(f"Saving paths map for {len(query_paths_map)} queries to {SAVE_PATH}...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(query_paths_map, f)
    print("Done! Memory efficient version.")

if __name__ == "__main__":
    main()