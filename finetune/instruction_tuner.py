import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import datasets
import psutil
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import Repository, create_repo
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    SchedulerType,
    get_scheduler,
)

logger = get_logger(__name__)

IGNORE_INDEX = -100
TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}



# Memory tracking utilities


def b2mb(x):
    """Convert bytes to megabytes."""
    return int(x / 2**20)


class TorchTracemalloc:
    """Context manager to track peak GPU and CPU memory usage."""

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()
        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """Return resident set size memory for the current process."""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)



# Data collator


@dataclass
class DataCollatorForInstructionTuning:
    """Collate examples for instruction tuning with label masking on prompts."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple(
            [torch.tensor(feature[key]) for feature in features]
            for key in ["input_ids", "attention_mask", "labels"]
        )
        input_ids = rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)



# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction-tune an auto-regressive language model "
                    "(Mistral, LLaMA-2, Falcon, GPT-2, Qwen2, etc.)"
    )

    # ---- Data ----
    parser.add_argument(
        "--load_data_from_disk",
        action="store_true",
        help="Load dataset from a local JSON file instead of the HF Hub",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        default="<your-dataset-name-or-path>",
        help="Dataset name on Hub or path to a local JSON file",
    )

    # ---- Model ----
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow executing remote code from the model repository",
    )
    parser.add_argument(
        "--hf_access_token",
        type=str,
        default="",
        help="HuggingFace access token for gated models",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model name on Hub or path to a local checkpoint",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to model_name_or_path)",
    )
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=4096,
        help="Sliding window size (used for Mistral models)",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float32", "float16", "bfloat16", "auto"],
        default="auto",
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="Use FlashAttention-2 (requires compatible GPU and package)",
    )

    # ---- BitsAndBytes quantization ----
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model with 8-bit quantization (LLM.int8())",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model with 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--llm_int8_threshold",
        type=float,
        default=6.0,
        help="Outlier threshold for LLM.int8()",
    )
    parser.add_argument(
        "--llm_int8_skip_modules",
        type=str,
        default=None,
        help="Comma-separated list of modules to skip in LLM.int8()",
    )
    parser.add_argument(
        "--llm_int8_enable_fp32_cpu_offload",
        action="store_true",
        help="Enable FP32 CPU offload in LLM.int8()",
    )
    parser.add_argument(
        "--llm_int8_has_fp16_weight",
        action="store_true",
        help="Run LLM.int8() with 16-bit main weights",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        choices=["float32", "float16", "bfloat16", "auto"],
        default=None,
        help="Compute dtype for 4-bit quantization",
    )
    parser.add_argument(
        "--bnb_4bit_quant_dtype",
        choices=["fp4", "nf4"],
        default="fp4",
        help="Quantization dtype for 4-bit (fp4 or nf4)",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        help="Use double quantization in 4-bit mode",
    )

    # ---- Preprocessing ----
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite cached training/evaluation sets",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=12,
        help="Number of workers for dataset preprocessing",
    )

    # ---- Optimization ----
    parser.add_argument(
        "--learning_rate",
        default=2.5e-5,
        type=float,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        default=None,
        help="Override total training steps (overrides num_train_epochs)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--adamw_fused",
        action="store_true",
        help="Use fused AdamW kernel (requires PyTorch >= 2.0)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_fraction",
        type=float,
        default=0.01,
        help="Fraction of total steps used for linear LR warmup",
    )

    # ---- Reproducibility / checkpoint resume ----
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume training from",
    )

    # ---- PEFT / LoRA ----
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Enable PEFT (LoRA) fine-tuning",
    )
    parser.add_argument(
        "--peft_lora_r",
        type=int,
        default=64,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--peft_lora_alpha",
        type=float,
        default=16,
        help="LoRA scaling factor (alpha)",
    )
    parser.add_argument(
        "--peft_lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability",
    )
    parser.add_argument(
        "--peft_target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names (no spaces)",
    )
    parser.add_argument(
        "--merge_weights",
        action="store_true",
        help="Merge LoRA weights into the base model after training",
    )

    # ---- Logging / saving ----
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Enable experiment tracking (WandB, TensorBoard, etc.)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            "Tracker(s) to report to: 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Only used when --with_tracking is set."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to write checkpoints and final model",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Save state every N steps, or 'epoch' to save after each epoch",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model checkpoints to the HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HF Hub repository name (defaults to output_dir basename)",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="HF Hub token for pushing (falls back to hf_access_token)",
    )
    parser.add_argument(
        "--private_repo",
        action="store_true",
        help="Create a private repository on the HF Hub",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot load model in both 8-bit and 4-bit mode simultaneously.")
    if args.push_to_hub and args.output_dir is None:
        raise ValueError("--output_dir must be set when using --push_to_hub.")

    return args



# Main training function


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory / Hub repository
    if accelerator.is_local_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id or Path(args.output_dir).absolute().name
            repo_id = create_repo(
                repo_name,
                exist_ok=True,
                token=args.hub_token,
                private=args.private_repo,
            ).repo_id
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                gitignore.write("step_*\nepoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    
    # Load dataset
    
    if args.load_data_from_disk:
        with open(args.dataset_name_or_path, "r") as f:
            raw_data = json.load(f)
        validation_split = (
            raw_data["validation"]
            if raw_data.get("validation")
            else raw_data["train"]
        )
        raw_dataset = DatasetDict({
            "train": Dataset.from_list(raw_data["train"]),
            "validation": Dataset.from_list(validation_split),
        })
    else:
        raw_dataset = load_dataset(
            args.dataset_name_or_path, token=args.hf_access_token
        )

    
    # BitsAndBytes configuration
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        llm_int8_threshold=args.llm_int8_threshold,
        llm_int8_skip_modules=(
            args.llm_int8_skip_modules.split(",")
            if args.llm_int8_skip_modules is not None
            else None
        ),
        llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
        llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        bnb_4bit_compute_dtype=(
            TORCH_DTYPES[args.bnb_4bit_compute_dtype]
            if args.bnb_4bit_compute_dtype is not None
            else None
        ),
        bnb_4bit_quant_dtype=args.bnb_4bit_quant_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    torch_dtype = TORCH_DTYPES[args.torch_dtype]

    
    # Load tokenizer
    
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    
    # Load base model
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
        use_flash_attention_2=args.use_flash_attention_2,
        quantization_config=bnb_config if (args.load_in_8bit or args.load_in_4bit) else None,
    )
    base_model.config.use_cache = False

    # Resize embeddings if the tokenizer vocabulary was extended
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        base_model.gradient_checkpointing_enable()

    
    # Preprocessing / tokenization
    
    raw_dataset_column_names = raw_dataset["train"].column_names

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The tokenizer supports a model_max_length longer than the default 1024. "
                "Override with --max_seq_length if you need longer sequences."
            )
        max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"--max_seq_length ({args.max_seq_length}) exceeds the model's maximum "
                f"({tokenizer.model_max_length}). Clamping to {tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        """
        Tokenize prompt+response pairs using the chat template.
        Labels are masked over the prompt portion so that loss is
        computed only on the response tokens.
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for prompt, response in zip(examples["inputs"], examples["targets"]):

            # 1. Build full chat-formatted sequence
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # 2. Tokenize the full sequence
            tokenized = tokenizer(
                full_text, truncation=True, max_length=max_seq_length
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 3. Tokenize prompt only to find the boundary
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokenized = tokenizer(
                prompt_text, truncation=True, max_length=max_seq_length
            )
            prompt_len = len(prompt_tokenized["input_ids"])

            # 4. Create labels: mask out prompt tokens
            labels = input_ids.copy()
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)

        return model_inputs

    preprocessed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=raw_dataset_column_names,
        desc="Tokenizing dataset",
    )

    train_dataset = preprocessed_dataset["train"]
    eval_dataset = preprocessed_dataset["validation"]

    # Log a few random training samples
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    # PEFT / LoRA setup
    
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=args.peft_lora_dropout,
            target_modules=args.peft_target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = base_model

    logger.info(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    
    # DataLoaders
    
    data_collator = DataCollatorForInstructionTuning(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
        num_workers=8,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
        num_workers=8,
    )

    # Prepare model with Accelerate before instantiating optimizer
    model = accelerator.prepare(model)

    
    # Optimizer and LR scheduler
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=args.adamw_fused,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.floor(args.lr_warmup_fraction * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate after Accelerate may have changed dataloader length
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Initialize experiment trackers
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("instruction_tuner", experiment_config)

    
    # Training loop
    
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples          = {len(train_dataset)}")
    logger.info(f"  Num epochs            = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total batch size      = {total_batch_size}")
    logger.info(f"  Grad accumulation     = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimizer steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if not checkpoint_path:
            dirs = sorted(
                [f.name for f in os.scandir(os.getcwd()) if f.is_dir()],
                key=os.path.getctime,
            )
            checkpoint_path = dirs[-1]

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(os.path.basename(checkpoint_path))[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):

        # ---- Train ----
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0 if args.with_tracking else None

            if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
            ):
                active_dataloader = accelerator.skip_first_batches(
                    train_dataloader, resume_step
                )
            else:
                active_dataloader = train_dataloader

            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if args.with_tracking:
                    accelerator.log(
                        {
                            "instant_loss": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        ckpt_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            ckpt_dir = os.path.join(args.output_dir, ckpt_dir)
                        accelerator.save_state(ckpt_dir)

                if completed_steps >= args.max_train_steps:
                    break

        # Log train memory
        accelerator.print(f"[Train] GPU memory (before / delta / peak): "
                          f"{b2mb(tracemalloc.begin)} / {tracemalloc.used} / {tracemalloc.peaked} MB")
        accelerator.print(f"[Train] CPU memory (before / delta / peak): "
                          f"{b2mb(tracemalloc.cpu_begin)} / {tracemalloc.cpu_used} / {tracemalloc.cpu_peaked} MB")

        # ---- Evaluate ----
        model.eval()
        losses = []
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(
                        accelerator.gather_for_metrics(
                            loss.repeat(args.per_device_eval_batch_size)
                        )
                    )

        # Log eval memory
        accelerator.print(f"[Eval]  GPU memory (before / delta / peak): "
                          f"{b2mb(tracemalloc.begin)} / {tracemalloc.used} / {tracemalloc.peaked} MB")
        accelerator.print(f"[Eval]  CPU memory (before / delta / peak): "
                          f"{b2mb(tracemalloc.cpu_begin)} / {tracemalloc.cpu_used} / {tracemalloc.cpu_peaked} MB")

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"Epoch {epoch}: perplexity={perplexity:.4f}  eval_loss={eval_loss:.4f}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # Mid-training Hub push
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress — epoch {epoch}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

        if args.checkpointing_steps == "epoch":
            ckpt_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                ckpt_dir = os.path.join(args.output_dir, ckpt_dir)
            accelerator.save_state(ckpt_dir)

    if args.with_tracking:
        accelerator.end_training()

    
    # Save final model
    
    if args.output_dir is not None:
        if not args.with_tracking:
            accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

    
    # Optional: merge LoRA weights into base model
    
    if args.use_peft and args.merge_weights:
        del base_model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir, device_map="auto", torch_dtype=torch_dtype
        )
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(
            output_merged_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_merged_dir)


if __name__ == "__main__":
    main()
