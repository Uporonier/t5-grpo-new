import os
from functools import partial
from typing import Optional, Union, Dict, Any, List

import torch
import trl
from trl import GRPOTrainer
from trl.trainer.utils import nanmin, nanmax
from accelerate.utils import gather_object
from datasets import Dataset
from tqdm.auto import tqdm
import shelve
from utils import convert_token_ids_to_key
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import  unwrap_model_for_generation
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from trl.data_utils import (
    apply_chat_template,
    prepare_multimodal_messages,
    )


from trl.trainer.utils import (

    entropy_from_logits,
    nanstd,
    pad,
    selective_log_softmax,
)

from trl.models.utils import  disable_gradient_checkpointing
# 辅助函数：构建简单 Trie
def build_simple_trie_on_the_fly(sequences):
    root = {}
    for seq in sequences:
        node = root
        for token in seq:
            if token not in node:
                node[token] = {}
            node = node[token]
    return root


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, ref_model=None, original_to_encoded_list=None, ce_loss_weight=0.1, *pos_args, **kwargs):
        # --- 1. 处理自定义参数 ---
        self.provided_ref_model = ref_model
        # 防止 kwargs 里还有 ref_model 传给父类导致报错
        if "ref_model" in kwargs:
            kwargs.pop("ref_model")

        self.original_to_encoded_list = original_to_encoded_list
        self.ce_loss_weight = ce_loss_weight
        self.custom_generation_kwargs = kwargs.pop("generation_kwargs", {})

        self.evaluator = kwargs.pop("evaluator", None)
        self.rank_map_path = kwargs.pop("rank_map_path", None)
        self.encoded_key_to_original = kwargs.pop("encoded_key_to_original", None)
        self.eval_generation_kwargs = kwargs.pop("eval_generation_kwargs", {})
        self.save_path = kwargs.pop("save_path", "")
        
        self.rank_db = None
        if self.rank_map_path:
            import shelve
            print(f"Opening Rank DB from {self.rank_map_path}...")
            self.rank_db = shelve.open(self.rank_map_path, flag='r')

        # --- 2. 核心逻辑：Beta Switch ---
        # 获取配置对象
        conf = kwargs.get("args") 
        if conf is None:
            raise ValueError("args (GRPOConfig) must be provided to CustomGRPOTrainer")
        
        # [关键] 保存原始 beta 值 (比如 0.1)
        real_beta = conf.beta 
        
        # [关键] 设为 0 以欺骗父类，跳过父类内部对 ref_model 的加载和处理
        conf.beta = 0.0 
        
        # --- 3. 调用父类初始化 ---
        # 注意：不要传 *pos_args，因为你上面把 args 变量名覆盖了，而且 GRPOTrainer 通常全是关键字参数
        super().__init__(**kwargs)

        # --- 4. 还原现场 & 注入模型 ---
        self.beta = real_beta       # 恢复 Trainer 内部属性
        conf.beta = real_beta       # 恢复 Config 对象属性
        self.ref_model = self.provided_ref_model # 注入模型

        # --- 5. [必须] 手动补课 (处理分布式和精度) ---
        # 因为欺骗了父类，父类没帮我们处理 ref_model，这里必须自己处理！
        if self.ref_model is not None and self.beta > 0:
            print(f"[CustomGRPOTrainer] Manually preparing ref_model with beta={self.beta}...")
            
            # 5.1 再次确保 Dropout 关闭 (双重保险)
            if conf.disable_dropout:
                from trl.trainer.utils import disable_dropout_in_model
                disable_dropout_in_model(self.ref_model)

            # 5.2 确保 Head 是 FP32 (数值稳定性)
            if conf.cast_lm_head_to_fp32:
                self._manual_cast_lm_head_to_fp32(self.ref_model)

            # 5.3 [至关重要] 使用 accelerator 包装模型
            # 如果不加这一步，多卡训练必崩，单卡也可能报错
            if self.is_deepspeed_enabled:
                from trl.models import prepare_deepspeed
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                from trl.models import prepare_fsdp
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                # 普通 DDP 或 单卡
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


        self.is_encoder_decoder = True

    def _manual_cast_lm_head_to_fp32(self, target_model):
        """父类闭包无法调用，手动复制一份"""
        def cast_inputs_to_fp32(module, inputs):
            if not inputs: return inputs
            return (inputs[0].to(torch.float32),) + inputs[1:]

        original_dtype = target_model.lm_head.weight.dtype
        target_model.lm_head = target_model.lm_head.float()
        target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

        if target_model.config.tie_word_embeddings:
            def cast_outputs(module, args, output):
                return output.to(original_dtype)
            target_model.model.embed_tokens.register_forward_hook(cast_outputs)
    
    # 析构函数：确保关闭数据库连接
    def __del__(self):
        if hasattr(self, 'rank_db') and self.rank_db:
            self.rank_db.close()

    def _generate_single_turn(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        #先去重 因为  unwrapped_model.generate（）对每个问题生成8个结果
        unique_prompts = prompts[::8]
        generate_inputs = self.processing_class(
                text=unique_prompts, padding=True, padding_side="right", return_tensors="pt"
            )
        # generate_inputs = super()._prepare_inputs(generate_inputs)

        # === 修复：手动移动到 GPU，而不是调用 _prepare_inputs ===
        generate_inputs = {
            k: v.to(self.accelerator.device) 
            for k, v in generate_inputs.items() 
            if isinstance(v, torch.Tensor)
        }
        # =======================================================

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config, disable_compile=True,**self.custom_generation_kwargs
            )


        # ======================= 【关键修改点】 =======================
        # 此时 generate_inputs['input_ids'] 是 64 行
        # 而 prompt_completion_ids 是 512 行
        # 为了让它们能 zip 在一起，我们需要把 输入的 64 行重新扩展回 512 行
        
        # 获取扩充倍数 (应该等于 8)
        repeat_factor = prompt_completion_ids.size(0) // generate_inputs['input_ids'].size(0)
        
        # 扩展 Prompt: (64, L) -> (512, L)
        # 这样 P1 就变成了 8 个 P1，刚好对应生成的 8 个 A1
        expanded_prompt_ids = generate_inputs["input_ids"].repeat_interleave(repeat_factor, dim=0)
        expanded_prompt_mask = generate_inputs["attention_mask"].repeat_interleave(repeat_factor, dim=0)
        
        


        prompt_length = generate_inputs["input_ids"].size(1)
        # =============================================================

        
        # Compute prompt length and extract completion ids
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)


        # completion_ids = prompt_completion_ids[:, prompt_length:]
        # === 修改开始：适配 T5 架构 ===
        config = getattr(self.model, "module", self.model).config
        if config.is_encoder_decoder:
            # T5: generate() 只返回生成的 completion，不需要切除 prompt
            completion_ids = prompt_completion_ids
        else:
            # Llama/Qwen: generate() 返回 prompt + completion，需要切除 prompt
            completion_ids = prompt_completion_ids[:, prompt_length:]
        # === 修改结束 ===

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]

        # 注意：这里我们要把 tensor 转回 list，这步不变
        prompt_ids = [p[m].tolist() for p, m in zip(expanded_prompt_ids, expanded_prompt_mask.bool())]

        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
        logprobs = None  # not used in this case
        extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields


    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        completion_ids=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            # [修改 2] 切分 completion_ids (如果传入了的话)
            completion_ids_batch = None
            if completion_ids is not None:
                completion_ids_batch = completion_ids[start : start + batch_size]

            # Build model inputs
            # [修改 3] 针对不同架构构建输入
            config = getattr(self.model, "module", self.model).config
            if config.is_encoder_decoder:
                # === T5 架构分支 ===
                if completion_ids_batch is None:
                    raise ValueError("For Encoder-Decoder models, `completion_ids` must be provided.")
                
                model_inputs = {
                    "input_ids": input_ids_batch,         # Encoder Input (Prompt)
                    "attention_mask": attention_mask_batch, 
                    "labels": completion_ids_batch        # Decoder Input (Completion)
                }
            else:
                # === Llama 架构分支 (原逻辑) ===
                # model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
                # # ... (原有的 logits_to_keep 处理逻辑仅对 Decoder-only 有效) ...
                # if "logits_to_keep" in self.model_kwarg_keys:
                #     model_inputs["logits_to_keep"] = logits_to_keep + 1

                 # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
                model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
                if image_grid_thw is not None and pixel_values is not None:
                    rows_per_image = image_grid_thw.prod(dim=-1)
                    rows_per_sample = torch.split(rows_per_image, num_images)
                    rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                    cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                    row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                    model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                    cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                    img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                    model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
                elif pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
                if pixel_attention_mask is not None:
                    model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
                if image_sizes is not None:
                    model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
                if token_type_ids is not None:
                    model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

                # Only add logits_to_keep if the model supports it
                if "logits_to_keep" in self.model_kwarg_keys:
                    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                    model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings
            # logits = model(**model_inputs).logits

            # Forward Pass
            outputs = model(**model_inputs)
            logits = outputs.logits

            # [修改 4] Logits 处理与对齐
            config = getattr(self.model, "module", self.model).config
            if config.is_encoder_decoder:
                # === T5 Logits 处理 ===
                # T5 输出的 logits 形状是 (Batch, Seq_Len_Labels, Vocab)
                # 且 logits[i] 对应的就是 labels[i]，无需移位
                
                # 确保 logits 覆盖了整个 completion (通常是的)
                # 如果有多余的，裁剪掉 (虽然一般不会有)
                if logits.size(1) > completion_ids_batch.size(1):
                     logits = logits[:, :completion_ids_batch.size(1), :]
                
                target_ids = completion_ids_batch
            else:
                # === Llama Logits 处理 (原逻辑) ===
                # 排除最后一个，因为是用来预测下一个token的
                logits = logits[:, :-1, :] 
                # 只保留最后 logits_to_keep 个
                logits = logits[:, -logits_to_keep:, :] 
                # 目标 ID 也是最后 logits_to_keep 个
                target_ids = input_ids_batch[:, -logits_to_keep:]

            # # Exclude the last value: it corresponds to the next token pred
            # logits = logits[:, :-1, :]  # (B, L-1, H)
            # # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            # logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # # Divide logits by sampling temperature.
            # # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            
            
            logits = logits / self.temperature
            # completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, target_ids)  # compute logprobs

            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies


    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        config = getattr(self.model, "module", self.model).config
        # 2. [关键修改] 根据架构决定如何构建 input_ids
        if config.is_encoder_decoder:
            # === T5 (Encoder-Decoder) 路径 ===
            # input_ids 仅包含 Encoder 的输入 (Prompt)
            # completion_ids 将独立传入用于计算 Loss
            input_ids = prompt_ids
            attention_mask = prompt_mask
        else:
            # === Llama (Decoder-only) 路径 ===
            # 原有逻辑：必须拼接 Prompt 和 Completion
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            completion_ids=completion_ids,# 必须显式传入 completion_ids，以便我们在 _get_per_token_logps_and_entropies 内部处理 T5 的 Labels
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        # 在 _compute_loss 开头加入
        if self.state.global_step < 2: # 只在前几步检查
            diff = (per_token_logps - old_per_token_logps) * completion_mask
            max_diff = diff.abs().max().item()
            print(f"🔍 [Step {self.state.global_step}] Max LogP Diff: {max_diff}")
            
            if max_diff > 1.0: # 容忍一点点浮点误差，但不能太大
                print("❌❌❌ 严重错误：模型没更新，LogP 却变了！这意味着输入数据对齐有问题！")
                # 打印错位的 LogP 看看
                print(f"Old: {old_per_token_logps[0, :5]}")
                print(f"New: {per_token_logps[0, :5]}")


        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    
    def _generate_and_score_completions(
            self, inputs: list[dict[str, torch.Tensor | Any]]
        ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)


        if self.is_encoder_decoder:
            cleaned_completion_ids_list = []
            for ids in completion_ids_list:
                # 转换为 list 以便操作
                curr_ids = list(ids) if not isinstance(ids, list) else ids
                # 循环移除开头的 0，直到第一个非 0 或者空
                while len(curr_ids) > 0 and curr_ids[0] == self.pad_token_id:
                    curr_ids = curr_ids[1:]
                cleaned_completion_ids_list.append(curr_ids)
            completion_ids_list = cleaned_completion_ids_list


        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="right")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="right")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        # prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

# === [New Feature] Calculate Local Ranks ===
        local_ranks_list = [] 
        rank_db = self.rank_db if self.rank_db else {}

        for i, gen_seq in enumerate(completion_ids_list):
            x = inputs[i]
            qid = str(x["qid"]) 
            query_prefix_map = rank_db.get(qid, {})
            
            sample_local_ranks = []
            current_prefix_tokens = [] 
            
            start_check_idx = 0
            if len(gen_seq) > 0 and gen_seq[0] == self.pad_token_id:
                sample_local_ranks.append(1)
                start_check_idx = 1
            
            is_off_track = False 

            for t_idx in range(start_check_idx, len(gen_seq)):
                token = gen_seq[t_idx]
                
                if token == self.eos_token_id or token == self.pad_token_id:
                    sample_local_ranks.append(1) 
                    is_off_track = True 
                    continue
                
                if is_off_track:
                    sample_local_ranks.append(101)
                    continue

                if len(current_prefix_tokens) == 0:
                    prefix_key = ""
                else:
                    prefix_key = convert_token_ids_to_key(current_prefix_tokens)
                
                next_token_ranks = query_prefix_map.get(prefix_key)
                
                if next_token_ranks and token in next_token_ranks:
                    rank = next_token_ranks[token]
                    sample_local_ranks.append(rank)
                    current_prefix_tokens.append(token)
                else:
                    sample_local_ranks.append(101)
                    is_off_track = True 
            
            local_ranks_list.append(sample_local_ranks)
            # 注入 inputs 供 calculate_rewards 使用
        for i, x in enumerate(inputs):
            x["local_ranks"] = local_ranks_list[i]
        # ===========================================
        # === [修改 1] 针对 T5 的分支逻辑 ===
        config = getattr(self.model, "module", self.model).config
        if config.is_encoder_decoder:
            # T5 路径: 
            # input_ids 就是 prompt_ids (Encoder 输入)
            # attention_mask 就是 prompt_mask (Encoder Mask)
            # completion_ids 保持独立，稍后作为 labels 传入
            model_input_ids = prompt_ids
            model_attention_mask = prompt_mask
        else:
            # Llama 路径 (Decoder-only): 
            # 必须拼接，input_ids 包含了整个序列
            model_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            model_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # =================================

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    

                    # prompt_completion_ids,
                    # attention_mask,
                    model_input_ids,       # T5: 只是 Prompt; Llama: 拼接结果
                    model_attention_mask,  # T5: 只是 Prompt Mask; Llama: 拼接 Mask


                    logits_to_keep,
                    batch_size,
                    completion_ids=completion_ids,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if not self.tools else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        # prompt_completion_ids,
                        # attention_mask,
                        model_input_ids,       # 同上
                        model_attention_mask,  # 同上
                        logits_to_keep,
                        batch_size=batch_size,
                        completion_ids=completion_ids,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            # prompt_completion_ids,
                            # attention_mask,
                            model_input_ids,       # 同上
                            model_attention_mask,  # 同上
                            logits_to_keep,
                            batch_size=batch_size,
                            completion_ids=completion_ids,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # 1. 准备参数
        reward_kwargs = {}
        if inputs:
            keys_to_gather = [k for k in inputs[0].keys() if k not in ["prompt", "completion"]]
            for k in keys_to_gather:
                # inputs 已经是扩展过的 (Batch * G)，直接取值即可
                reward_kwargs[k] = [x[k] for x in inputs]

        # 注入 local_ranks
        reward_kwargs["local_ranks"] = local_ranks_list

        # rewards_list_per_sample = self._calculate_rewards(inputs, prompts_text, completions_text, completion_ids_list)
        rewards_list_per_sample = self.reward_funcs[0](
                    prompts=prompts_text,
                    completions=completions_text,
                    completion_ids=completion_ids_list,
                    **reward_kwargs
                )


        # ==============================================================================
        # 【新增 2】Token-Level Advantage 计算
        # ==============================================================================
        
        # 1. Padding: 将 Reward 列表填充为 Tensor (Batch*G, SeqLen)
        padded_rewards_list = []
        for r_seq, c_mask in zip(rewards_list_per_sample, completion_mask):
            r_tensor = torch.tensor(r_seq, device=device, dtype=torch.float32)
            # Pad 到与 completion_ids 对齐
            if len(r_tensor) < c_mask.shape[0]:
                r_tensor = torch.nn.functional.pad(r_tensor, (0, c_mask.shape[0] - len(r_tensor)), value=0.0)
            elif len(r_tensor) > c_mask.shape[0]:
                r_tensor = r_tensor[:c_mask.shape[0]]
            # Masking
            r_tensor = r_tensor * c_mask.float()
            padded_rewards_list.append(r_tensor)
            
        # Shape: (Batch*G, SeqLen)
        rewards = torch.stack(padded_rewards_list)

        # 2. Group Normalization (Token-Level)
        # 我们要计算每个 Token 位置在 Group 内的优势
        # B_total, L = rewards.shape
        # G = self.num_generations
        
        # # Shape: (Batch, G, SeqLen)
        # grouped_rewards = rewards.view(B_total // G, G, L)
        
        # # 沿着 Group 维度 (dim=1) 计算均值和方差
        # # 这样对于同一个 Prompt 的第 t 个 token，我们比较 G 个生成结果在该位置的得分
        # group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        # group_std = grouped_rewards.std(dim=1, keepdim=True)
        
        # # 计算优势
        # grouped_advantages = (grouped_rewards - group_mean) / (group_std + 1e-4)
        
        # # 展平回 (Batch*G, SeqLen)
        # advantages = grouped_advantages.view(B_total, L)
        # advantages = advantages * completion_mask.float()


        # 1. 准备数据形状
        # rewards: (B_total, L) -> (Batch, G, L)
        B_total, L = rewards.shape
        G = self.num_generations
        
        grouped_rewards = rewards.view(B_total // G, G, L)
        grouped_masks = completion_mask.view(B_total // G, G, L)

        # 2. 计算每个时间步，组内有多少个有效样本 (Valid Count)
        # sum(dim=1) 沿着组维度求和
        # shape: (Batch, 1, L)
        valid_counts = grouped_masks.sum(dim=1, keepdim=True)
        
        # 防止除以 0 (虽然 mask 为 0 时分子也是 0，但为了数值稳定，设最小值为 1)
        valid_counts_safe = valid_counts.clamp(min=1.0)

        # 3. 计算 Masked Mean (只对有效部分求和，除以有效数量)
        # 注意：rewards 在 mask=0 的地方已经是 0.0 了，所以直接 sum 即可
        group_sum = grouped_rewards.sum(dim=1, keepdim=True)
        group_mean = group_sum / valid_counts_safe

        # 4. 计算 Masked Std
        # 方差公式: sum((x - mean)^2 * mask) / count
        # 我们需要把 pad 部分的 (0 - mean)^2 剔除掉，所以要乘 grouped_masks
        diff = grouped_rewards - group_mean
        diff_sq = (diff.pow(2)) * grouped_masks # 关键：mask 掉 padding 的方差贡献
        group_var = diff_sq.sum(dim=1, keepdim=True) / valid_counts_safe
        group_std = group_var.sqrt()

        # 5. 计算优势
        # 如果 group_std 为 0 (比如只有一个样本有效)，优势应该为 0
        grouped_advantages = (grouped_rewards - group_mean) / (group_std + 0.1)

        # 进行 Clamping，防止极端值影响训练稳定性
        grouped_advantages = torch.clamp(grouped_advantages, min=-5.0, max=5.0)
        
        # 6. 再次 Mask
        # 此时 padding 位置的 advantage 可能是无意义的数值 (因为我们刚才为了防除0 clamp 了 count)
        # 必须再次乘以 mask 确保 padding 处为 0
        grouped_advantages = grouped_advantages * grouped_masks

        # 7. 展平回 (Batch*G, SeqLen)
        advantages = grouped_advantages.view(B_total, L)
        
    # ==============================================================================


        scalar_rewards = rewards.sum(dim=1) # 简单的总分用于查看
        grouped_scalar_rewards = scalar_rewards.view(-1, G)
        self._metrics[mode]["reward"].append(grouped_scalar_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(grouped_scalar_rewards.std(dim=1).mean().item())
        
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        if self.reward_func_names:
            self._logs["rewards"][self.reward_func_names[0]].extend(scalar_rewards.tolist())


        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if self.tools:
            output["tool_mask"] = tool_mask
        return output

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            if self.evaluator is None or self.encoded_key_to_original is None:
                print("警告: 未提供 `evaluator` 等必要参数，跳过自定义评估。")
                return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # 1. 准备数据集
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Evaluation requires an eval_dataset.")
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

            # =========================================================================
            # 【关键修复 Step 1】: 确保数据集包含唯一索引列
            # =========================================================================
            # 我们给数据集加一列 "_eval_index"，用于后续的多卡去重
            # 检查是否已有索引列，如果没有则添加
            if "_eval_index" not in eval_dataset.column_names:
                # map操作开销很小，因为它不处理大张量，只是加个int
                eval_dataset = eval_dataset.map(
                    lambda _, idx: {"_eval_index": idx}, 
                    with_indices=True,
                    desc="Adding index for consistent evaluation"
                )

            from torch.utils.data import DataLoader

            # 2. 构造 DataLoader
            # 使用 simple_collate_fn，让 batch 返回 List[Dict]
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False, 
                drop_last=False,
                collate_fn=simple_collate_fn, # 确保这个函数在文件顶部已定义
                num_workers=2
            )
            
            # 让 Accelerator 处理分布式采样 (DistributedSampler)
            # 注意：这里会自动对数据进行切分，并不整除时会进行 Padding (重复采样)
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.eval()

            # 用于存储 (index, prediction, truth) 的临时列表
            local_results = []
            
            # 3. 评估循环
            if self.accelerator.is_main_process:
                print(f"\n开始评估 (进程数: {self.accelerator.num_processes})...")
                
            for step, batch in tqdm(
                enumerate(eval_dataloader), 
                total=len(eval_dataloader), 
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                # batch 是 List[Dict]，手动解包
                # 注意：这里我们同时获取 prompt, truth 和 刚才注入的 _eval_index
                prompts = [x["prompt"] for x in batch]
                truths = [x["ground_truth_docids"] for x in batch]
                indices = [x["_eval_index"] for x in batch] # 获取唯一ID

                # Tokenize
                inputs = self.processing_class(
                    text=prompts, return_tensors="pt", padding=True,
                     truncation=True, add_special_tokens=True,
                )
                inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

                # Generate (Beam Search)
                with torch.no_grad():
                    generated_ids = unwrapped_model.generate(
                        **inputs, 
                        **self.eval_generation_kwargs, # 确保包含 num_beams, max_length 等
                        disable_compile=True
                    )
                
                # Post-process (对齐 Script B 逻辑)
                num_beams = self.eval_generation_kwargs.get("num_beams", 1)
                docid_format = "msmarco"
                lookup_fallback = True

                # 遍历当前 Batch 内的每个样本
                for i in range(len(prompts)):
                    sample_idx = indices[i] # 当前样本的全局唯一索引
                    sample_truth = set(truths[i])
                    
                    ranked_list = []
                    # 获取该样本对应的所有 Beams
                    prompt_outputs = generated_ids[i * num_beams : (i + 1) * num_beams]
                    
                    for output_ids in prompt_outputs:
                        # 解码
                        key_primary = docid2string_msmarco(output_ids.cpu().tolist())
                        
                        # 准备 Fallback Key
                        if docid_format == "msmarco":
                            alt = key_primary[:-2] if key_primary.endswith(",1") else key_primary + ",1"
                        else:
                            alt = key_primary + ",1" if key_primary and not key_primary.endswith(",1") else key_primary[:-2]

                        # 查表 (假设 safe_lookup 已导入)
                        found_docids = safe_lookup(
                            key_primary, 
                            self.encoded_key_to_original, 
                            fallback=lookup_fallback, 
                            alt_key=alt
                        )
                        
                        # 去重添加到结果列表
                        for doc_id in found_docids:
                            if doc_id and doc_id not in ranked_list:
                                ranked_list.append(doc_id)
                    
                    # 【关键】将 (Index, Preds, Truth) 三元组存入列表
                    local_results.append({
                        "index": sample_idx,
                        "preds": ranked_list,
                        "truth": sample_truth
                    })

            # 4. 收集所有进程的结果
            # gather_object 会将所有进程的列表拼接在一起 (注意：这是乱序且包含重复 Padding 的)
            all_results_gathered = gather_object(local_results)

            # =========================================================================
            # 【关键修复 Step 2】: 基于 Index 进行去重和排序
            # =========================================================================
            # DistributedSampler 为了对齐 Batch Size，会复制部分样本。
            # 直接截断会导致数据错位。我们必须用字典根据 index 去重。
            
            unique_results_map = {}
            for item in all_results_gathered:
                idx = item["index"]
                # 存入字典，如果 idx 重复，覆盖即可（内容是一样的）
                unique_results_map[idx] = {
                    "preds": item["preds"],
                    "truth": item["truth"]
                }
            
            # 验证数据完整性
            total_unique_samples = len(unique_results_map)
            
            # 按照 Index 从 0 到 N-1 排序，恢复原始数据集顺序
            sorted_indices = sorted(unique_results_map.keys())
            
            final_preds_list = [unique_results_map[i]["preds"] for i in sorted_indices]
            final_truth_list = [unique_results_map[i]["truth"] for i in sorted_indices]

            # 5. 计算指标 (所有进程都计算一遍，结果完全一致)
            # 注意：这里 evaluator 接收的是去重且排序后的完整列表
            if len(final_preds_list) > 0:
                results = self.evaluator.evaluate_ranking(final_truth_list, final_preds_list)
            else:
                results = {} # 防止空数据集报错

            # 提取标量用于打印
            _mrr10 = results.get('MRR@10', 0.0)
            if hasattr(_mrr10, 'item'): _mrr10 = _mrr10.item()
                
            _mrr = results.get('MRR', 0.0)
            if hasattr(_mrr, 'item'): _mrr = _mrr.item()

            _r1 = results.get('R@1', 0.0)
            if hasattr(_r1, 'item'): _r1 = _r1.item()

            _r5 = results.get('R@5', 0.0)
            if hasattr(_r5, 'item'): _r5 = _r5.item()

            _r10 = results.get('R@10', 0.0)
            if hasattr(_r10, 'item'): _r10 = _r10.item()
            
            _r100 = results.get('R@100', 0.0)
            if hasattr(_r100, 'item'): _r100 = _r100.item()

            # 仅主进程打印日志
            if self.accelerator.is_main_process:
                print(f"\n[Eval Result] Count: {total_unique_samples}")
                print(f"mrr@10: {_mrr10:.4f}, mrr: {_mrr:.4f}")
                print(f"r@1: {_r1:.4f}, r@5: {_r5:.4f}, r@10: {_r10:.4f}, r@100: {_r100:.4f}")
                
                # 将 numpy/tensor 转为 python float 供 logger 使用
                metrics_for_log = {f"{metric_key_prefix}_{k}": (v.item() if hasattr(v, 'item') else v) for k, v in results.items()}
                self.log(metrics_for_log)

            # 返回 metrics 字典 (所有进程都需要返回，防止 Trainer 挂起)
            metrics_with_prefix = {f"{metric_key_prefix}_{k}": (v.item() if hasattr(v, 'item') else v) for k, v in results.items()}
            return metrics_with_prefix


def simple_collate_fn(features):
    """
    一个简单的 collator，不做任何堆叠，直接返回 List[Dict]。
    """
    return features



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
    fallback: bool = True,
    alt_key: str = None,
) -> List[str]:
    """
    Script B 的核心逻辑：带 Fallback 的查表
    """
    # 辅助函数：统一返回 list
    def _to_list(val):
        if isinstance(val, list): return val
        return [val]

    if key in table:
        return _to_list(table[key])
    
    if fallback and alt_key is not None and alt_key in table:
        return _to_list(table[alt_key])
    
    return [] # 没找到返回空列表