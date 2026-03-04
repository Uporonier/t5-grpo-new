import os
from tqdm import tqdm

# === 1. 配置路径 ===
SOURCE_FILES = [
    "/data1/fengjun/workspace/GR-GRPO/QWen3-embedding-test/data_filter/new_encode_and_rank/4b-msmarco-4096-dev/msmarco-4096-top100-dev.tsv",
    # "/data1/fengjun/workspace/GR-GRPO/GR-RL/data/filtered_ranked_results_dev.tsv"
]

# 指定新的输出目录
OUTPUT_DIR = "/data1/fengjun/workspace/ly/ddro/masmarco-top100"

def process_ranking_file(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"❌ 原文件不存在: {file_path}")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建新的输出文件路径
    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(output_dir, file_name)

    print(f"\n正在处理: {file_name}")
    print(f"   -> 读取自: {file_path}")
    print(f"   -> 保存到: {new_file_path}")
    
    modified_count = 0
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(new_file_path, 'w', encoding='utf-8') as f_out:
        
        # 处理表头
        header = f_in.readline()
        f_out.write(header)

        for line in tqdm(f_in, desc=f"Processing {file_name}"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            # 格式应该是: qid \t relevant_docs \t ranked_docs
            if len(parts) != 3:
                f_out.write(line + '\n')
                continue

            qid = parts[0]
            relevant_docs_str = parts[1]
            ranked_docs_str = parts[2]

            # 解析列表
            relevant_docs = relevant_docs_str.split(',') if relevant_docs_str else []
            ranked_docs = ranked_docs_str.split(',') if ranked_docs_str else []

            # 如果没有 GT，则无法置顶，原样写入
            if not relevant_docs:
                f_out.write(line + '\n')
                continue

            # 取第一个 GT 作为目标置顶文档
            target_gt = relevant_docs[0] 
            
            # === 核心逻辑: 强制置顶 ===
            is_modified = False
            
            # 检查第一个位置是否已经是该 GT
            if ranked_docs and ranked_docs[0] == target_gt:
                pass # 已经是第一了，无需操作
            else:
                is_modified = True
                if target_gt in ranked_docs:
                    # 情况 1: GT 在列表中，但不是第一 -> 移到第一
                    ranked_docs.remove(target_gt)
                    ranked_docs.insert(0, target_gt)
                else:
                    # 情况 2: GT 不在列表中 -> 插到第一
                    ranked_docs.insert(0, target_gt)
                    # 保持长度限制 (维持 100)
                    if len(ranked_docs) > 100:
                        ranked_docs = ranked_docs[:100]

            if is_modified:
                modified_count += 1
            
            total_count += 1
            
            # 重新组合并写入
            new_ranked_str = ",".join(ranked_docs)
            f_out.write(f"{qid}\t{relevant_docs_str}\t{new_ranked_str}\n")

    print(f"✅ 完成! {file_name} 已处理。")
    print(f"   共 {total_count} 条，修正顺序 {modified_count} 条。")

if __name__ == "__main__":
    for f in SOURCE_FILES:
        process_ranking_file(f, OUTPUT_DIR)