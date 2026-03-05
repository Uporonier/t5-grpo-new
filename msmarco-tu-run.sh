# export CUDA_LAUNCH_BLOCKING=1

PRETRAIN_MODEL_PATH=/data2/chenran/workspace/ddro/pretrain/t5-base
CHECKPOINT_PATH=/data2/chenran/workspace/ly/workspace/models/ddro-sft/ddro-msmarco-tu-sft
# CHECKPOINT_PATH=/data2/chenran/workspace/ddro/checkpoint/ddro-msmarco-tu
# PRETRAIN_MODEL_PATH=/data1/fengjun/workspace/models/google-t5/t5-large
# CHECKPOINT_PATH=/data1/fengjun/workspace/ly/ddro/outputs-ddro/hf_model_tu_finetuned
TRAIN_QUERIES_FILE=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries.tsv.gz
# TRAIN_QUERIES_FILE=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries-debug.tsv.gz

TRAIN_RANKINGS_FILE=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-top100-4096/msmarco-4096-top100-train.tsv


DEV_QUERIES_FILES=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/final_filtered_queries_dev.tsv.gz
DEV_RANKINGS_FILE=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-top100-4096/msmarco-4096-top100-dev.tsv
QRELS_FILE_PATH=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-docdev-qrels.tsv.gz

# DEV_QUERIES_FILES=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-queries.tsv.gz
# DEV_RANKINGS_FILE=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-top100-4096/msmarco-4096-top100-train.tsv
# QRELS_FILE_PATH=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/msmarco-doctrain-qrels.tsv.gz

# ENCODED_DOCID_PATH=/data2/chenran/workspace/ly/workspace/modelscope/msmarco-tu-datasets/url_title_docid.txt
ENCODED_DOCID_PATH=/data2/chenran/workspace/ddro/src/pretrain/resources/datasets/processed/msmarco-data/encoded_docid/tu_msmarco_docids.txt

# export CUDA_VISIBLE_DEVICES=6
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=5
SAVE_PATH=/data2/chenran/workspace/ly/workspace/grpo_ddro/msmarco-tu-new-t5-grpo/outputs/$(date +%Y%m%d-%H%M%S)
mkdir -p ${SAVE_PATH}

# LAUNCH_CMD="python -m debugpy --listen 9501 --wait-for-client"
# LAUNCH_CMD="torchrun --nproc_per_node=8 --nnodes=1 --master_port=29500"
# LAUNCH_CMD="accelerate launch  --num_processes=1  --main_process_port 29680"
# LAUNCH_CMD="python"


LAUNCH_CMD="accelerate launch --num_processes=1 --main_process_port 29680"

export SWANLAB_PROJECT="GRPO-T5-NEW-DEBUG"
# CUDA_VISIBLE_DEVICES=0,1,2,3  ${LAUNCH_CMD}  ./train_grpo.py \
CUDA_VISIBLE_DEVICES=0  ${LAUNCH_CMD}  /data2/chenran/workspace/ly/workspace/grpodebug/msmarco-tu-new-t5-grpo/train_grpo.py \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --train_queries_file ${TRAIN_QUERIES_FILE} \
    --dev_queries_file ${DEV_QUERIES_FILES} \
    --encoded_docid_path ${ENCODED_DOCID_PATH} \
    --train_rankings_file ${TRAIN_RANKINGS_FILE} \
    --dev_rankings_file ${DEV_RANKINGS_FILE} \
    --qrels_file_path ${QRELS_FILE_PATH} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_generations 8 \
    --beta 0.0 \
    --save_steps 800 \
    --eval_steps 800 \
    # --report_to swanlab 2>&1 | tee ${SAVE_PATH}/train_$(date +%Y%m%d-%H%M%S).log
    # --eval_on_start False \

