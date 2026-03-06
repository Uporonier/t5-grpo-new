
import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from datasets import Dataset, disable_caching
import transformers
import trl
from trl import GRPOConfig # type: ignore
from T5GRPOTrainer import CustomGRPOTrainer
from t5_grpo_trainer import Seq2SeqGRPOTrainer
from rewarder import RewardScorer
from utils import (
    convert_token_ids_to_key, 
    load_generative_retrieval_model, 
    load_encoded_docids_and_create_map, 
    create_dataset_with_ranking_list,
    build_partial_trie,
    load_qrels,
)

from evaluate import evaluator
import shelve
os.environ["WANDB_MODE"] = "disabled"
os.environ["SWANLAB_DISABLED"] = "false"


print(f"Using TRL library version: {trl.__version__}")

# 禁用Hugging Face数据集缓存
disable_caching()

# try:
#     import debugpy
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Debugpy failed to start: {e}")

from transformers import TrainerCallback  # <--- 记得导入这个

class GlobalRealTimeCallback(TrainerCallback):
    def __init__(self, reward_scorer, print_every_steps=10):
        """
        Args:
            reward_scorer: 你的计分器实例
            print_every_steps: 每多少步汇总打印一次 (建议设为 10 或 50)
        """
        self.reward_scorer = reward_scorer
        self.print_every_steps = print_every_steps


def main():
    parser = argparse.ArgumentParser(description="Train a T5-based Generative Retrieval model using GRPO")
    parser.add_argument("--train_queries_file", type=str, default="data/msmarco-data/msmarco-doctrain-queries.tsv.gz")
    parser.add_argument("--train_rankings_file", type=str, default="data/ranked_results_with_qrels_and_top100_docs.tsv")
    parser.add_argument("--dev_queries_file", type=str, default="data/msmarco-data/msmarco-docdev-queries.tsv.gz")
    parser.add_argument("--dev_rankings_file", type=str, default="data/ranked_results_with_qrels_and_top100_docs_dev.tsv")
    parser.add_argument("--qrels_file_path", type=str, default="data/qrels/msmarco-docdev-qrels.txt.gz")
    parser.add_argument("--encoded_docid_path", type=str, default="data/encode_docid/url_title_docid.txt")
    parser.add_argument("--pretrain_model_path", type=str, default="pretrain/t5-base")
    parser.add_argument("--checkpoint_path", type=str, default="models/ddro_models/ddro_ms_tu_model_final.pkl")
    parser.add_argument("--trie_path", type=str, default="data/tire/docid_trie.pkl", help="Path to the pre-built docID Trie pickle file.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--add_doc_num", type=int, default=6144)
    parser.add_argument("--use_origin_head", type=bool, default=True)
    parser.add_argument("--max_prompt_length", type=int, default=32)
    parser.add_argument("--max_completion_length", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for generation. If None, use greedy decoding.")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for data loading. Defaults to all CPUs.")
    parser.add_argument("--report_to", type=str, default="none", help="Where to report logs (e.g., 'tensorboard', 'wandb').")
    args = parser.parse_args()

    print(args)

    transformers.logging.set_verbosity_info()

    # RANK_MAP_PATH = "/home/aizoo/data/workspace/ly/ddro/datasets/msmarco-tu-localrank-correct/query_prefix_rank_map_db"
    
    
    RANK_MAP_PATH = "/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-map-localrank-correct-4096/query_prefix_rank_map_db"
    # --- 模型加载 ---
    print("Loading models...")
    model, tokenizer = load_generative_retrieval_model(args)


    
     # --- 数据加载 ---
    encoded_key_to_original, original_to_encoded_list, all_encoded = load_encoded_docids_and_create_map(args.encoded_docid_path)

    rank_db = shelve.open(RANK_MAP_PATH, flag='r')
    # --- 奖励函数 ---
    reward_scorer = RewardScorer(encoded_key_to_original, original_to_encoded_list, 
                                 gamma=0.9,
                                 rank_db=rank_db
                    )
    
    print(f"Building Trie on-the-fly...")
    trie_sequences = [[tokenizer.pad_token_id] + item for item in all_encoded]
    docid_trie = build_partial_trie(trie_sequences)
        
    def prefix_allowed_tokens_fn(batch_id, sent):
        sent_list = sent.tolist()
        outputs = docid_trie.get(sent_list)
        if len(outputs) == 0:
            outputs = [tokenizer.eos_token_id]
        return outputs

    # 调用时传入 max_samples 参数
    print("Creating training dataset...")
    train_dataset = create_dataset_with_ranking_list(
        args.train_queries_file, 
        args.train_rankings_file, 
        max_samples=None # 使用命令行参数
    )
    print(train_dataset[0])
    
    print("Creating evaluation dataset...")
    dev_qrels = load_qrels(args.qrels_file_path)
    dev_dataset = create_dataset_with_ranking_list(
        args.dev_queries_file,
        args.dev_rankings_file,
        qrels_map=dev_qrels,
        max_samples=None # 使用命令行参数
    )
    print(dev_dataset[0])

    print(f"Train dataset size: {len(train_dataset)}, Dev dataset size: {len(dev_dataset)}")

    myevaluator = evaluator()
    model_init_kwargs = {
            "device_map": None,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,
        }
    # --- GRPO 配置 ---
    grpo_config = GRPOConfig(
        model_init_kwargs=model_init_kwargs,
        num_iterations= 1,
        half_precision_backend=False,
        ddp_find_unused_parameters=False,
        # gradient_checkpointing=False,

        output_dir=args.output_dir, 
        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        # dataloader_num_workers=32,
        # dataloader_prefetch_factor=16,
        
        logging_steps=args.logging_steps, 
        report_to=args.report_to,
        log_completions=False,
        
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps", 
        save_steps=args.save_steps, 
        save_total_limit=2,

        load_best_model_at_end=True, 
        metric_for_best_model="eval_R@1", 
        greater_is_better=True,

        # bf16=torch.cuda.is_bf16_supported(), 
        # fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False, 
        beta=args.beta, 
        
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length, 
        max_completion_length=args.max_completion_length,
        
        seed=42,
    )
    
    # --- CustomGRPOTrainer 实例化 ---
    trainer = Seq2SeqGRPOTrainer(
        # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn,
        token_level_rewards=True,
        beam_search=True,
        model=model,
        args=grpo_config,
        reward_funcs=[reward_scorer.reward_function_rank_agnostic],
        train_dataset=train_dataset, 
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        
        evaluator=myevaluator,
        encoded_key_to_original=encoded_key_to_original,
        eval_generation_kwargs={
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "num_beams": 15,
            "num_return_sequences": 15, # 返回与束宽相同数量的序列
            "do_sample": False, # 关闭采样
            "max_length": 100,
        },
    )

    # trainer.valid_docid_check = prefix_allowed_tokens_fn

    # 假设你的 args.logging_steps 是 10 或 100
    logging_freq = args.logging_steps if args.logging_steps > 0 else 10

    # 注册准实时监控 Callback
    realtime_callback = GlobalRealTimeCallback(reward_scorer, print_every_steps=logging_freq)
    trainer.add_callback(realtime_callback)

    # --- 训练和保存 ---
    # print(model.config)
    # trainer.evaluate()
    trainer.train()

    eval_output = trainer.evaluate()
    print(eval_output)

    # final_save_path = os.path.join(args.output_dir, "grpo_model_final.pkl")

    # unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    # torch.save(unwrapped_model.state_dict(), final_save_path)
    # print(f"Final model saved to {final_save_path}")

    # trainer.end_of_training()

if __name__ == "__main__":
    main()