import os
import gzip
import torch
import sys
import math
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer

# ==========================================
# 1. 环境与路径配置 (解决 ModuleNotFoundError)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 确保从父目录导入 utils 里的工具函数
from utils import build_partial_trie, load_encoded_docids_and_create_map, docid2string_msmarco, safe_lookup, load_qrels

# ==========================================
# 2. 模型加载与词表对齐工具
# ==========================================
def align_tokenizer_vocab_with_model(tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    target_vocab_size = model.config.vocab_size
    current_tokenizer_size = len(tokenizer)
    
    if current_tokenizer_size < target_vocab_size:
        num_to_add = target_vocab_size - current_tokenizer_size
        print(f"📊 词表不匹配检测: 模型({target_vocab_size}) > Tokenizer({current_tokenizer_size})")
        print(f"正在添加 {num_to_add} 个占位符 Token...")
        new_tokens = [f"<extra_token_{i}>" for i in range(num_to_add)]
        tokenizer.add_tokens(new_tokens)
        # 注意：如果是 SFT 后的模型，通常词表已经扩充过，这里是为了防止意外
    return tokenizer

def load_generative_retrieval_model(path):
    # 使用 use_fast=True 提高处理速度
    tokenizer = T5Tokenizer.from_pretrained(path, use_fast=True)
    base_model = T5ForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.float32,
        device_map=None, # 由脚本手动控制 device
        low_cpu_mem_usage=False
    )
    tokenizer = align_tokenizer_vocab_with_model(tokenizer, base_model)
    return base_model, tokenizer

# ==========================================
# 3. 筛选配置
# ==========================================
CHECKPOINT_PATH = "/data2/chenran/workspace/ly/workspace/models/ddro-sft/ddro-msmarco-tu-sft"
ENCODED_DOCID_PATH = "/data2/chenran/workspace/ddro/src/pretrain/resources/datasets/processed/msmarco-data/encoded_docid/tu_msmarco_docids.txt"
ORIGINAL_TRAIN_QUERIES = "/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries.tsv.gz"
# 筛选结果保存路径
OUTPUT_FILTERED_QUERIES = "/data2/chenran/workspace/ly/workspace/grpodebug/msmarco-tu-new-t5-grpo/train_datasets_split/msmarco-doctrain-queries-top15.tsv.gz"
# 训练集 Qrels 用于判定 GT 排名
TRAIN_QRELS_PATH = "/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-qrels.tsv.gz"

TOP_K_THRESHOLD = 15
BATCH_SIZE = 64  # 根据显存 24G-80G 灵活调整

# ==========================================
# 4. 核心逻辑：获取并保存 Top-15 命中的数据集
# ==========================================
def get_filtered_dataset():
    # 强制指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型并移动到 GPU
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model, tokenizer = load_generative_retrieval_model(CHECKPOINT_PATH)
    model.to(device)
    model.eval()

    # 2. 加载 DocID 映射表和构建 Trie
    print("Loading DocID maps and building Trie...")
    encoded_key_to_original, _, all_encoded = load_encoded_docids_and_create_map(ENCODED_DOCID_PATH)
    
    # 构建 Trie 用于 prefix_allowed_tokens_fn
    trie_sequences = [[tokenizer.pad_token_id] + item for item in all_encoded]
    docid_trie = build_partial_trie(trie_sequences)

    # 定义约束函数
    def prefix_allowed_tokens_fn(batch_id, sent):
        # 将 sent (tensor) 转为 list，确保 Trie 查询兼容性
        outputs = docid_trie.get(sent.tolist())
        return outputs if len(outputs) > 0 else [tokenizer.eos_token_id]

    # 3. 读取数据：训练查询和 Qrels
    queries = []
    with gzip.open(ORIGINAL_TRAIN_QUERIES, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                queries.append(parts)

    print(f"Loading Qrels from {TRAIN_QRELS_PATH}...")
    train_qrels = load_qrels(TRAIN_QRELS_PATH)

    # 4. 推理与筛选
    filtered_count = 0
    print(f"Starting filtering... Target: Top-{TOP_K_THRESHOLD} Hits")
    
    # 
    
    with gzip.open(OUTPUT_FILTERED_QUERIES, "wt", encoding="utf-8") as out_f:
        # 分 batch 进行推理以提高速度
        for i in tqdm(range(0, len(queries), BATCH_SIZE)):
            batch_data = queries[i : i + BATCH_SIZE]
            batch_qids = [x[0] for x in batch_data]
            batch_texts = [x[1] for x in batch_data]

            # 编码输入并确保在 GPU 上
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                # 修复核心：显式指定 input_ids, attention_mask 和 decoder 相关的 ID
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,
                    num_beams=TOP_K_THRESHOLD,
                    num_return_sequences=TOP_K_THRESHOLD,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id
                )

            # 遍历 Batch 结果
            for j, qid in enumerate(batch_qids):
                gt_set = train_qrels.get(qid, set())
                if not gt_set:
                    continue

                # 从生成的 outputs 中切分出当前 sample 的所有 beams (TOP_K 个)
                start_idx = j * TOP_K_THRESHOLD
                end_idx = (j + 1) * TOP_K_THRESHOLD
                sample_beams = outputs[start_idx:end_idx]

                found_hit = False
                for beam_ids in sample_beams:
                    # 1. ID 序列转 key 字符串
                    pred_key = docid2string_msmarco(beam_ids.tolist())
                    # 2. Key 转 DocID (带 fallback 逻辑适配你的数据库)
                    pred_docids = safe_lookup(
                        pred_key, 
                        encoded_key_to_original, 
                        fallback=True, 
                        alt_key=pred_key + ",1"
                    )
                    
                    # 3. 命中判定
                    if any(d in gt_set for d in pred_docids):
                        found_hit = True
                        break
                
                # 如果前 15 名里有任意一个命中了真实文档，保留该 Query
                if found_hit:
                    out_f.write(f"{qid}\t{batch_texts[j]}\n")
                    filtered_count += 1

    print(f"✅ 筛选完成！原始样本: {len(queries)}, 保留样本: {filtered_count}")
    print(f"输出文件: {OUTPUT_FILTERED_QUERIES}")

if __name__ == "__main__":
    get_filtered_dataset()