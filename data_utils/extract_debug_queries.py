import gzip
import csv
import os

# 1. 定义文件路径
source_path = "/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries.tsv.gz"
output_path = "/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries-debug.tsv.gz"

# 2. 定义你要查找的目标 Query 列表 (使用集合 set 提高查找速度)
# 注意：我根据你的日志提取了所有不重复的 Prompt 内容
target_prompts = {
    "what to wear to a gala event",
    "what do bubbles in my urine mean",
    "exhort meaning",
    "cast of character on dobie gillis",
    "how many presidents in us history have been impeached?",
    "fixed cost per unit formula",
    "average public college president salary",
    "which of the following is an example of energy conservation",
    "to unblock caller id",
    "causes of anemia in pregnancy",
    "heritage federal cu routing number",
    "average public college president salary",
    "can hormones affect cholesterol"


}

def extract_bad_cases():
    print(f"开始处理...")
    print(f"源文件: {source_path}")
    print(f"目标文件: {output_path}")
    
    found_count = 0
    
    # 使用 gzip 打开输入和输出文件
    # 'rt' = read text (自动处理解码), 'wt' = write text
    with gzip.open(source_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
        
        # TSV 读取器
        reader = csv.reader(f_in, delimiter='\t')
        # TSV 写入器
        writer = csv.writer(f_out, delimiter='\t')
        
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
                
            # msmarco query 文件通常格式: [query_id, query_text]
            query_id = row[0]
            query_text = row[1]
            
            # 检查当前行的 query_text 是否在目标列表中
            # strip() 用于去除可能存在的首尾空格
            if query_text.strip() in target_prompts:
                writer.writerow(row)
                found_count += 1
                # 打印一下找到的内容，方便确认
                print(f"找到 [{found_count}]: ID={query_id}, Text={query_text}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"共找到并写入 {found_count} 行数据到 {output_path}")

if __name__ == "__main__":
    extract_bad_cases()