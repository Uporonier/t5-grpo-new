
"""
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes=6 \
/data2/chenran/workspace/ly/workspace/grpodebug/msmarco-tu-new-t5-grpo/train_datasets_split/datasets_split.py
"""
import os
import gzip
import torch
import sys
from tqdm import tqdm
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer
# try:
#     import debugpy
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Debugpy failed to start: {e}")

# ==========================================
# 1. 环境与路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import build_partial_trie, load_encoded_docids_and_create_map, docid2string_msmarco, safe_lookup, load_qrels

def align_tokenizer_vocab_with_model(tokenizer, model):
    target_vocab_size = model.config.vocab_size
    current_tokenizer_size = len(tokenizer)
    if current_tokenizer_size < target_vocab_size:
        num_to_add = target_vocab_size - current_tokenizer_size
        new_tokens = [f"<extra_token_{i}>" for i in range(num_to_add)]
        tokenizer.add_tokens(new_tokens)
    return tokenizer

def load_generative_retrieval_model(path):
    tokenizer = T5Tokenizer.from_pretrained(path, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=False
    )
    tokenizer = align_tokenizer_vocab_with_model(tokenizer, model)
    return model, tokenizer


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


# 临时文件夹用于存放断点数据
TEMP_DIR = "/data2/chenran/workspace/ly/workspace/grpodebug/msmarco-tu-new-t5-grpo/train_datasets_split/temp_splits"
os.makedirs(TEMP_DIR, exist_ok=True)

TOP_K_THRESHOLD = 15
BATCH_SIZE = 128  

# ==========================================
# 3. 核心逻辑
# ==========================================
def get_filtered_dataset():
    accelerator = Accelerator()
    device = accelerator.device

    # 1. 加载模型与 Trie
    if accelerator.is_main_process:
        print(f"Loading model on {accelerator.num_processes} GPUs...")
    
    model, tokenizer = load_generative_retrieval_model(CHECKPOINT_PATH)
    encoded_key_to_original, _, all_encoded = load_encoded_docids_and_create_map(ENCODED_DOCID_PATH)
    
    trie_sequences = [[tokenizer.pad_token_id] + item for item in all_encoded]
    docid_trie = build_partial_trie(trie_sequences)

    def prefix_allowed_tokens_fn(batch_id, sent):
        return docid_trie.get(sent.tolist()) or [tokenizer.eos_token_id]

    # 2. 准备数据与断点恢复逻辑
    queries = []
    with gzip.open(ORIGINAL_TRAIN_QUERIES, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2: queries.append(parts)
    
    train_qrels = load_qrels(TRAIN_QRELS_PATH)

    # 每个进程维护自己的临时保存文件
    rank = accelerator.process_index
    temp_file_path = os.path.join(TEMP_DIR, f"rank_{rank}_hits.tsv")
    processed_qids_path = os.path.join(TEMP_DIR, f"rank_{rank}_processed.log")

    # 加载已经处理过的 qid 列表 (断点续传)
    processed_qids = set()
    if os.path.exists(processed_qids_path):
        with open(processed_qids_path, "r") as f:
            processed_qids = set(line.strip() for line in f)

    # 将数据切分到不同的 GPU 进程
    with accelerator.split_between_processes(queries) as process_queries:
        # 过滤掉本进程已经处理过的查询
        remaining_queries = [q for q in process_queries if q[0] not in processed_qids]
        
        # 以追加模式打开文件
        with open(temp_file_path, "a") as hit_f, open(processed_qids_path, "a") as proc_f:
            
            # 3. 推理循环
            for i in tqdm(range(0, len(remaining_queries), BATCH_SIZE), 
                          disable=not accelerator.is_local_main_process,
                          desc=f"GPU {rank} Processing"):
                
                batch_data = remaining_queries[i : i + BATCH_SIZE]
                batch_qids = [x[0] for x in batch_data]
                batch_texts = [x[1] for x in batch_data]

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = model.to(device).generate(
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

                # 处理结果
                for j, qid in enumerate(batch_qids):
                    gt_set = train_qrels.get(qid, set())
                    if not gt_set: 
                        proc_f.write(f"{qid}\n") # 即使没GT也算处理过
                        continue

                    start_idx = j * TOP_K_THRESHOLD
                    sample_beams = outputs[start_idx : (j + 1) * TOP_K_THRESHOLD]

                    found_hit = False
                    for beam_ids in sample_beams:
                        pred_key = docid2string_msmarco(beam_ids.tolist())
                        pred_docids = safe_lookup(pred_key, encoded_key_to_original, fallback=True, alt_key=pred_key + ",1")
                        if any(d in gt_set for d in pred_docids):
                            found_hit = True
                            break
                    
                    if found_hit:
                        hit_f.write(f"{qid}\t{batch_texts[j]}\n")
                        hit_f.flush() # 实时刷入磁盘
                    
                    proc_f.write(f"{qid}\n")
                    proc_f.flush()

    # 4. 等待所有进程结束，由主进程汇总
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Merging all temporary results...")
        final_results = set()
        for r in range(accelerator.num_processes):
            t_file = os.path.join(TEMP_DIR, f"rank_{r}_hits.tsv")
            if os.path.exists(t_file):
                with open(t_file, "r") as f:
                    for line in f:
                        final_results.add(line.strip())

        with gzip.open(OUTPUT_FILTERED_QUERIES, "wt", encoding="utf-8") as out_f:
            for item in sorted(list(final_results)):
                out_f.write(item + "\n")
        
        print(f"✅ All Done. Final count: {len(final_results)}")
        print(f"File saved at: {OUTPUT_FILTERED_QUERIES}")

if __name__ == "__main__":
    get_filtered_dataset()