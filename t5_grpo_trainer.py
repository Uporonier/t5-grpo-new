import os
from typing import Optional, Union, Dict, Any, List
from tqdm.auto import tqdm

import torch
from datasets import Dataset

from accelerate import logging
from accelerate.utils import gather

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.trainer.grpo_trainer import pad, selective_log_softmax, entropy_from_logits, gather_object
from trl.models.utils import disable_gradient_checkpointing, unwrap_model_for_generation

from utils import docid2string_msmarco, safe_lookup

logger = logging.get_logger(__name__)
# 定义在文件顶层，这样 pickle 就能通过模块路径找到它
def simple_collate_fn(batch):
    return batch

class Seq2SeqGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer adapted for encoder-decoder (seq2seq) models like T5.

    Overrides three methods that assume decoder-only architecture:
    1. _get_per_token_logps_and_entropies: feeds encoder_input_ids and decoder_input_ids
       separately instead of concatenating [prompt, completion].
    2. _generate_and_score_completions: handles T5 generate() returning only decoder tokens.
    3. _compute_loss: same encoder/decoder separation for the training forward pass.
    """

    # beam_search: Whether to use beam search instead of sampling for generation
    # prefix_allowed_tokens_fn = None  # Optional function for constraining generation with beam search
    # evaluator = None  # Optional evaluator for custom evaluation logic

    def __init__(self, *, beam_search=False, token_level_rewards=False, prefix_allowed_tokens_fn=None, evaluator=None, encoded_key_to_original=None, eval_generation_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.beam_search = beam_search
        self.token_level_rewards = token_level_rewards
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.evaluator = evaluator
        self.encoded_key_to_original = encoded_key_to_original
        self.eval_generation_kwargs = eval_generation_kwargs or {}

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep,
        batch_size=None, compute_entropy=False, **kwargs
    ):
        """
        For encoder-decoder: input_ids is the ENCODER input (prompt).
        We need decoder_input_ids from kwargs to compute logps over the completion.

        Key difference from decoder-only:
        - Decoder-only: model([prompt+completion]) -> logits, then shift by 1
        - Encoder-decoder: model(encoder_input=prompt, decoder_input=shift_right(completion))
          -> logits directly aligned with completion tokens (no shift needed)
        """
        decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)

        if decoder_input_ids is None:
            raise ValueError(
                "Seq2SeqGRPOTrainer requires decoder_input_ids in "
                "_get_per_token_logps_and_entropies"
            )

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            enc_ids = input_ids[start:end]
            enc_mask = attention_mask[start:end]
            dec_ids = decoder_input_ids[start:end]
            dec_mask = decoder_attention_mask[start:end] if decoder_attention_mask is not None else None

            model_inputs = {
                "input_ids": enc_ids,
                "attention_mask": enc_mask,
                "decoder_input_ids": dec_ids,
                "use_cache": False,
            }
            if dec_mask is not None:
                model_inputs["decoder_attention_mask"] = dec_mask

            logits = model(**model_inputs).logits  # (B, dec_len, vocab)
            # For encoder-decoder, logits[i] predicts the token at position i of the
            # target. decoder_input_ids is shift_right(completion_ids), so logits[i]
            # predicts completion_ids[i]. No shift needed — but we drop the last logit
            # if decoder_input_ids is one longer than completion_ids.
            # We keep only logits_to_keep positions from the end.
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            # The target tokens are the last logits_to_keep tokens of the actual
            # completion (not the shifted decoder_input_ids).
            target_ids = kwargs.get("completion_ids_for_logps", None)
            if target_ids is not None:
                target_batch = target_ids[start:end]
            else:
                # Fallback: derive from decoder_input_ids by shifting left
                target_batch = dec_ids[:, 1:]
                target_batch = target_batch[:, -logits_to_keep:]

            logps = selective_log_softmax(logits, target_batch)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(entropy_from_logits(logits))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _build_decoder_input_ids(self, completion_ids):
        """
        Build decoder_input_ids by prepending the decoder start token and dropping
        the last token of completion_ids (standard teacher-forcing shift).
        T5 uses pad_token_id as the decoder start token.
        """
        model = self.accelerator.unwrap_model(self.model)
        decoder_start = model.config.decoder_start_token_id
        bsz = completion_ids.size(0)
        start_tokens = completion_ids.new_full((bsz, 1), decoder_start)
        # decoder_input_ids = [decoder_start, comp_0, comp_1, ..., comp_{n-2}]
        # This aligns so that logits[i] predicts completion_ids[i]
        decoder_input_ids = torch.cat([start_tokens, completion_ids[:, :-1]], dim=1)
        return decoder_input_ids

    def _generate_and_score_completions(self, inputs):
        """
        Override to handle encoder-decoder generation and scoring.

        Key differences from the parent:
        - T5 generate() returns only decoder output, not [prompt + completion]
        - We store prompt_ids and completion_ids separately (not concatenated)
        - Log-prob computation passes them to encoder and decoder respectively
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # Generate completions
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Pad prompt and completion tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # Mask truncated completions
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Build decoder_input_ids for log-prob computation
        decoder_input_ids = self._build_decoder_input_ids(completion_ids)
        decoder_attention_mask = completion_mask.clone()
        # The start token position should always be attended to
        decoder_attention_mask = torch.cat([
            torch.ones(decoder_attention_mask.size(0), 1, device=device, dtype=torch.long),
            decoder_attention_mask[:, :-1],
        ], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        # Compute old log-probs (for importance sampling)
        with torch.no_grad(), disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_ids, prompt_mask, logits_to_keep,
                    batch_size=batch_size,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    completion_ids_for_logps=completion_ids,
                )
            else:
                old_per_token_logps = None

            # Reference model log-probs for KL penalty
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_ids, prompt_mask, logits_to_keep,
                        batch_size=batch_size,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        completion_ids_for_logps=completion_ids,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, prompt_ids, prompt_mask, logits_to_keep,
                            batch_size=batch_size,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            completion_ids_for_logps=completion_ids,
                        )
            else:
                ref_per_token_logps = None

        # Decode for reward computation
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Calculate rewards
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        # print(f"Rewards per function: {rewards_per_func}")
        if self.token_level_rewards:
            # token-level reward: (batch_size, num_reward_funcs, max_len)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0).unsqueeze(0)).nansum(dim=1) # (batch_size, max_len)
            # Compute advantages (group-normalized)
            num_generations = (
                self.num_generations if mode == "train" else self.num_generations_eval
            )
            mean_grouped_rewards = rewards.view(-1, num_generations, rewards.shape[-1]).mean(dim=1) # (num_groups, max_len)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0) # (batch_size, max_len)
            advantages = rewards - mean_grouped_rewards # (batch_size, max_len)

            if self.scale_rewards in ["group", "none"]:
                std_rewards = rewards.view(-1, num_generations, rewards.shape[-1]).std(dim=1) # (num_groups, max_len)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0) # (batch_size, max_len)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards)
            else:
                raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)   # reward_std很小  e 从1e-4改成0.1

            # Slice to local process
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            all_process_advantages = advantages.clone()
            advantages = advantages[process_slice]
            #####新加的这个切片是为了适配 T5 的生成长度（24 或 28 或 31），而不是之前全局的 112。
            # 必须把长度从全局的 112 切回本地的 24 (或 28, 31)
            advantages = advantages[:, :completion_ids.size(1)]
            #############################################################################
            # Log metrics
            for i, reward_func_name in enumerate(self.reward_func_names):
                self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                    torch.nanmean(rewards_per_func[:, :, i]).sum().item()
                )
            self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
            self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
            self._metrics[mode]["frac_reward_zero_std"].append(
                is_std_zero.float().mean().item()
            )

            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

            output = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "advantages": advantages,
                "num_items_in_batch": num_items_in_batch,
            }
            if old_per_token_logps is not None:
                output["old_per_token_logps"] = old_per_token_logps
            if ref_per_token_logps is not None:
                output["ref_per_token_logps"] = ref_per_token_logps
            return output
        
        else:
            # sample-level reward
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1) # (batch_size, 1)

            # Compute advantages (group-normalized)
            num_generations = (
                self.num_generations if mode == "train" else self.num_generations_eval
            )
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1) # (num_groups, 1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0) # (batch_size, 1)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards in ["group", "none"]:
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards)
            else:
                raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)  #1e-4

            # Slice to local process
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            all_process_advantages = advantages.clone()
            advantages = advantages[process_slice]

            # Log metrics
            for i, reward_func_name in enumerate(self.reward_func_names):
                self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                    torch.nanmean(rewards_per_func[:, i]).item()
                )
            self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
            self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
            self._metrics[mode]["frac_reward_zero_std"].append(
                is_std_zero.float().mean().item()
            )

            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

            output = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "advantages": advantages,
                "num_items_in_batch": num_items_in_batch,
            }
            if old_per_token_logps is not None:
                output["old_per_token_logps"] = old_per_token_logps
            if ref_per_token_logps is not None:
                output["ref_per_token_logps"] = ref_per_token_logps
            return output

    def _compute_loss(self, model, inputs):
        """
        Override to pass encoder/decoder inputs separately instead of concatenating.
        The GRPO loss math itself is identical — only the log-prob computation changes.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        decoder_attention_mask = inputs["decoder_attention_mask"]
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, prompt_ids, prompt_mask, logits_to_keep,
            compute_entropy=True,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            completion_ids_for_logps=completion_ids,
        )

        # Advantages
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None
            else old_per_token_logps
        )

        # Importance sampling ratio
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # Clipped surrogate loss (same as parent)
        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Loss aggregation
        mask = completion_mask
        if self.loss_type in ["grpo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            clip_ratio = (
                (coef_1 > 1 + self.epsilon_high) | (coef_1 < 1 - self.epsilon_low)
            ).float().mean()
            self._metrics[mode]["clip_ratio"].append(
                self.accelerator.gather(clip_ratio).mean().item()
            )

        return loss
    
    def _generate_single_turn(self, prompts):
        mode = "train" if self.model.training else "eval"
        if self.beam_search and mode == "train":
            return self._generate_single_turn_beam_search(prompts)
        else:
            return super()._generate_single_turn(prompts)

    def _generate_single_turn_beam_search(self, prompts):
        """
        Override to handle T5's generate() which returns only decoder output tokens,
        not the concatenated [prompt + completion] that decoder-only models return.

        Uses beam search instead of sampling to ensure unique outputs within each group.
        The incoming `prompts` are already duplicated by RepeatSampler:
            [p0, p0, ..., p0, p1, p1, ..., p1, ...]
        We deduplicate, run beam search with num_beams=num_generations, and each beam
        returns num_generations unique sequences per prompt.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # --- Step 1: Deduplicate prompts ---
        # Take every num_generations-th prompt to get unique ones
        unique_prompts = prompts[::num_generations]

        # Tokenize unique prompts only
        generate_inputs = self.processing_class(
            text=unique_prompts, padding=True, padding_side="left", return_tensors="pt"
        )
        generate_inputs = super(GRPOTrainer, self)._prepare_inputs(generate_inputs)

        # --- Step 2: Beam search generation ---
        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            # Override generation config for beam search
            from transformers import GenerationConfig
            beam_gen_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                num_beams=num_generations,
                num_return_sequences=num_generations,
                # prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                do_sample=False,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                decoder_start_token_id=unwrapped_model.config.decoder_start_token_id,
            )
            # Input: (unique_prompt_num, seq_len)
            # Output: (unique_prompt_num * num_generations, dec_len)
            generated_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=beam_gen_config,
            )

        # --- Step 3: Strip decoder_start_token_id ---
        model_cfg = self.accelerator.unwrap_model(self.model).config
        if generated_ids.size(1) > 0 and generated_ids[0, 0].item() == model_cfg.decoder_start_token_id:
            completion_ids = generated_ids[:, 1:]
        else:
            completion_ids = generated_ids

        # --- Step 4: Build prompt_ids matching the output layout ---
        # generate() with num_return_sequences=N returns N consecutive rows per input.
        # So output is already [p0_beam0, p0_beam1, ..., p1_beam0, p1_beam1, ...]
        # which matches the expected GRPO layout exactly.
        enc_ids = generate_inputs["input_ids"]       # (unique_prompt_num, enc_len)
        enc_mask = generate_inputs["attention_mask"]  # (unique_prompt_num, enc_len)
        # Repeat each prompt's ids to match the num_generations beams
        prompt_ids_expanded = enc_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask_expanded = enc_mask.repeat_interleave(num_generations, dim=0)

        # --- Step 5: Mask everything after first EOS in completions ---
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        comp_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # Convert to lists (matching parent's expected format)
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids_expanded, prompt_mask_expanded.bool())]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, comp_mask.bool())]
        logprobs = None
        extra_fields = {}

        return prompt_ids, completion_ids, logprobs, extra_fields
    
    def evaluate(self,
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

            # 我们给数据集加一列 "_eval_index"，用于后续的多卡去重
            # 检查是否已有索引列，如果没有则添加
            if "_eval_index" not in eval_dataset.column_names:
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
                # collate_fn=lambda x: x,
                collate_fn=simple_collate_fn,
                num_workers=4
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
                ground_truth_original_docids = [x["ground_truth_original_docids"] for x in batch]
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
                
                # Post-process
                num_beams = self.eval_generation_kwargs.get("num_beams", 1)
                docid_format = "msmarco"

                # 遍历当前 Batch 内的每个样本
                for i in range(len(prompts)):
                    sample_idx = indices[i] # 当前样本的全局唯一索引
                    sample_truth = set(ground_truth_original_docids[i])
                    
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
                            fallback=True, 
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

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if self.token_level_rewards:
            return self._calculate_rewards_token_level(inputs, prompts, completions, completion_ids_list)
        else:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _calculate_rewards_token_level(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        local_max_len = max(len(ids) for ids in completion_ids_list)
        global_max_len = max(gather_object([local_max_len])) # get global max_len across all processes

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), global_max_len, device=device) # (batch_size, num_reward_funcs, max_len)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                ) # (batch_size, current_sample_completion_len)

                for j, rewards in enumerate(output_reward_func):
                    # Convert None values to NaN
                    rewards = [r if r is not None else torch.nan for r in rewards]
                    rewards_per_func[j, i, :len(rewards)] = torch.tensor(rewards, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # print(f"{device}: rewards_per_func before gather: {rewards_per_func.shape}")
        rewards_per_func = gather(rewards_per_func) # (total_batch_size, num_reward_funcs, max_len)
        return rewards_per_func

        
    # def _calculate_rewards_token_level(self, inputs, prompts, completions, completion_ids_list):
    #     device = self.accelerator.device
        
    #     # 1. 确定当前 batch 中 completion 的最大长度
    #     max_len = max(len(ids) for ids in completion_ids_list) if completion_ids_list else 0
        
    #     # 2. 初始化奖励张量 (注意：这是本地进程的 size)
    #     # shape: (batch_size_per_device, max_completion_len, num_reward_funcs)
    #     rewards_per_func = torch.zeros(len(prompts), max_len, len(self.reward_funcs), device=device)

    #     # 准备奖励函数的参数
    #     keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
    #     reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    #     reward_kwargs["trainer_state"] = self.state

    #     # 3. 循环调用奖励函数
    #     for i, (reward_func, _, reward_func_name) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
    #     ):
    #         with profiling_context(self, reward_func_name):
    #             # 调用你的奖励函数，它应该返回一个 List[List[float]]
    #             # 外层长度 = len(prompts)，内层长度 = 各个 completion 的长度
    #             output_reward_func = reward_func(
    #                 prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
    #             )

    #             # 4. 将奖励填入张量
    #             for j, rewards in enumerate(output_reward_func):
    #                 # 如果奖励是 None，转为 NaN
    #                 rewards = [r if r is not None else torch.nan for r in rewards]
    #                 # 填入对应的 token 位置
    #                 rel_len = min(len(rewards), max_len)
    #                 rewards_per_func[j, :rel_len, i] = torch.tensor(
    #                     rewards[:rel_len], dtype=torch.float32, device=device
    #                 )

    #     # 5. 【关键】直接返回本地计算的结果，绝对不要调用 gather(rewards_per_func)
    #     # TRL 基类会自动处理后续的跨进程均值/标准差计算
    #     return rewards_per_func