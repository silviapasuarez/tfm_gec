#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento para corrección gramatical en gallego con Salamandra-7B-Instruct (ChatML)

Dataset esperado en JSONL con campos:
{
  "id": ...,
  "prompt": "...",
  "incorrect": "...",
  "correct": "...",
  "categories": [...],
  "changes": [{"original": "...", "incorrect": "...", "op": "replace"|"delete"|"insert"}, ...],
}

Salida objetivo SIEMPRE:
  ### Corrección: <frase_corrixida>
  ### Cambios: <orig1 → inc1 (op); orig2 → inc2 (op); ...>
"""

from unsloth import FastLanguageModel, is_bfloat16_supported, to_sharegpt
from unsloth.chat_templates import (
    standardize_sharegpt,
    get_chat_template,
    train_on_responses_only,
)

import torch, os
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# ================== CONFIG ==================
MODEL_NAME   = "/workspace/corrector_salamandra_stage1" # "BSC-LT/salamandra-7b-instruct"
DATASET_PATH = "/workspace/full_dataset_b_kept_thr3.0_augmented.jsonl"
OUTPUT_DIR   = "/workspace/corrector_salamandra_stage2"

TRAIN_PHASE  = "stage2"   # "stage1" -> pérdida solo en respuesta, "stage2" -> pérdida completa
SEED         = 3407

MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT   = False   # A100 80GB → puedes dejar False y usar bf16
DTYPE          = None    # Unsloth lo detecta

BATCH_SIZE     = 4
GRAD_ACCUM     = 4
EPOCHS         = 3
LR             = 5e-6
WARMUP_STEPS   = 100
WEIGHT_DECAY   = 0.00

MAX_CHANGES_LIST = 10

GENERAL_SYSTEM = (
"""Es un modelo especializado en corrixir exclusivamente os erros gramaticais e ortográficos en galego.\n
Non debes cambiar nin engadir ningunha palabra se xa está correcta.\n
O texto de saída debe ser idéntico ao orixinal, salvo nas correccións estritamente necesarias.\n
Responde sempre en galego.\n\nSaída:\n\n'## Corrección: <texto>'\n'## Cambios: <lista>'."""
)

# ================== MODELO ==================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

# Configuración LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

# Tokens básicos (Salamandra usa SentencePiece tipo LLaMA)
tokenizer.pad_token = "<unk>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.unk_token_id = tokenizer.unk_token_id

# ================== DATASET ==================
raw = load_dataset("json", data_files=DATASET_PATH, split="train")
splits = raw.train_test_split(test_size=0.1, seed=SEED)
train_raw, eval_raw = splits["train"], splits["test"]

def _fmt_change(c: dict) -> str:
    orig = (c.get("original") or "").strip()
    inc  = (c.get("incorrect") or "").strip()
    op   = (c.get("op") or "").strip()
    if not orig and not inc: return ""
    return f"{inc} → {orig} ({op})" if op else f"{inc} → {orig}"

def _fmt_changes_list(ex: dict, limit: int = MAX_CHANGES_LIST) -> str:
    changes = ex.get("changes") or []
    items = [s for s in (_fmt_change(c) for c in changes[:limit]) if s]
    if not items: return ""
    extra = len(changes) - len(items)
    return "; ".join(items) + (f"; +{extra} máis" if extra > 0 else "")

def _build_io(example: dict) -> dict:
    prompt    = example.get("prompt") or "Corrixe a seguinte oración:"
    incorrect = example["incorrect"]
    correct   = example["correct"].strip()
    input_txt = f"{prompt}\n\n{incorrect}"

    changes_str = _fmt_changes_list(example)

    # Siempre incluímos Cambios, aunque esté vacío
    output_txt = f"### Corrección: {correct}\n\n### Cambios: "
    if changes_str:
        output_txt += changes_str
    else:
        output_txt += "(ningún cambio atopado)"

    output_txt += tokenizer.eos_token

    return {"SYSTEM": GENERAL_SYSTEM, "INPUT": input_txt, "OUTPUT": output_txt}


train_tmp = train_raw.map(_build_io)
eval_tmp  = eval_raw.map(_build_io)

MERGED_PROMPT = "[[ {SYSTEM}\n\n]]{INPUT}"

train_sg = to_sharegpt(train_tmp, merged_prompt=MERGED_PROMPT, output_column_name="OUTPUT")
eval_sg  = to_sharegpt(eval_tmp,  merged_prompt=MERGED_PROMPT, output_column_name="OUTPUT")

train_sg = standardize_sharegpt(train_sg)
eval_sg  = standardize_sharegpt(eval_sg)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

def _to_text(batch):
    key = "messages" if "messages" in batch else "conversations"
    return {"text": [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in batch[key]]}

train_proc = train_sg.map(_to_text, batched=True, remove_columns=train_sg.column_names)
eval_proc  = eval_sg.map(_to_text,  batched=True, remove_columns=eval_sg.column_names)

print("Ejemplo renderizado:\n", train_proc[0]["text"][:400], "...\n")

# ================== TRAINING ==================
bf16_flag = is_bfloat16_supported()     # A100 → True
fp16_flag = not bf16_flag

args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs = EPOCHS,
    warmup_steps = WARMUP_STEPS,
    learning_rate = LR,
    weight_decay = WEIGHT_DECAY,
    lr_scheduler_type = "linear",
    logging_steps = 10,
    save_strategy = "epoch",
    eval_strategy = "epoch",
    save_total_limit = 1,
    report_to = "none",
    fp16 = fp16_flag,
    bf16 = bf16_flag,
    optim = "adamw_torch",
    seed = SEED,
    max_grad_norm = 1.0,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_proc,
    eval_dataset = eval_proc,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    args = args,
    packing=False,
)

if TRAIN_PHASE == "stage1":
    trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant"
    )

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Entrenamiento {TRAIN_PHASE} finalizado. Pesos guardados en: {OUTPUT_DIR}")
