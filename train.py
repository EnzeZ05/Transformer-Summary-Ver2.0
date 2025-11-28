import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
CACHE_DIR = "/root/autodl-fs/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer

MODEL_NAME = "gpt2-large"

D_MODEL = 1280
NUM_LAYERS = 36
NUM_HEADS = 20
D_FF = 5120

MAX_SEQ_LEN = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
SAVE_DIR = "summarizer"
USE_AMP = True
DROPOUT = 0.1

DATA_DIR = "/root/autodl-tmp"
TRAIN_JSON = os.path.join(DATA_DIR, "english_train.jsonl")


def load_gpt2_weights(my_model, hf_model):
    print("Start copying GPT-2 Large weights (Conv1D -> Linear)...")

    my_model.token_emb.weight.data.copy_(hf_model.transformer.wte.weight.data)
    my_model.pos_emb.weight.data.copy_(hf_model.transformer.wpe.weight.data)

    for i in range(len(my_model.layers)):
        my_block = my_model.layers[i]
        hf_block = hf_model.transformer.h[i]

        c_attn_w = hf_block.attn.c_attn.weight.data.t()
        c_attn_b = hf_block.attn.c_attn.bias.data

        w_q, w_k, w_v = c_attn_w.split(D_MODEL, dim=0)
        b_q, b_k, b_v = c_attn_b.split(D_MODEL, dim=0)

        my_block.attention.wq.weight.data.copy_(w_q)
        my_block.attention.wq.bias.data.copy_(b_q)

        my_block.attention.wk.weight.data.copy_(w_k)
        my_block.attention.wk.bias.data.copy_(b_k)

        my_block.attention.wv.weight.data.copy_(w_v)
        my_block.attention.wv.bias.data.copy_(b_v)

        my_block.attention.wo.weight.data.copy_(hf_block.attn.c_proj.weight.data.t())
        my_block.attention.wo.bias.data.copy_(hf_block.attn.c_proj.bias.data)

        my_block.ln_1.weight.data.copy_(hf_block.ln_1.weight.data)
        my_block.ln_1.bias.data.copy_(hf_block.ln_1.bias.data)

        my_block.ln_2.weight.data.copy_(hf_block.ln_2.weight.data)
        my_block.ln_2.bias.data.copy_(hf_block.ln_2.bias.data)

        my_block.feed_forward.w1.weight.data.copy_(hf_block.mlp.c_fc.weight.data.t())
        my_block.feed_forward.w1.bias.data.copy_(hf_block.mlp.c_fc.bias.data)

        my_block.feed_forward.w2.weight.data.copy_(hf_block.mlp.c_proj.weight.data.t())
        my_block.feed_forward.w2.bias.data.copy_(hf_block.mlp.c_proj.bias.data)

    my_model.ln_f.weight.data.copy_(hf_model.transformer.ln_f.weight.data)
    my_model.ln_f.bias.data.copy_(hf_model.transformer.ln_f.bias.data)

    print("GPT-2 Large weights loaded successfully!")


def preprocess_function(examples, tokenizer):
    if "article" in examples:
        docs = examples["article"]
        sums = examples["highlights"]
    else:
        text_key = "text" if "text" in examples else "Text"
        sum_key = "summary" if "summary" in examples else "Summary"
        docs, sums = examples[text_key], examples[sum_key]

    input_ids_list = []
    labels_list = []

    for doc, summ in zip(docs, sums):
        prompt = f"Summarize:\n{doc}\nSummary:"

        sum_ids = tokenizer.encode(
            summ + tokenizer.eos_token,
            add_special_tokens=False,
            )

        max_prompt_len = MAX_SEQ_LEN - len(sum_ids)
        if max_prompt_len <= 10:
            continue

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )

        ids = prompt_ids + sum_ids
        labels = [-100] * len(prompt_ids) + sum_ids

        input_ids_list.append(ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
        labels = [torch.tensor(ex["labels"]) for ex in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading HF Model: {MODEL_NAME} from {CACHE_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    print("Initializing Custom ModernDecoder (GPT-2 Large Architecture)...")
    my_model = transformer.ModernDecoder(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        layers=NUM_LAYERS,
        heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        token_id=tokenizer.pad_token_id,
    ).to(device)

    load_gpt2_weights(my_model, hf_model)
    del hf_model

    if device.type == "cuda":
        my_model.to(dtype=torch.bfloat16)
        torch.cuda.empty_cache()

    raw_datasets = load_dataset("json", data_files={"train": TRAIN_JSON})
    train_dataset = raw_datasets["train"].select(range(5000)).map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(my_model.parameters(), lr=LEARNING_RATE)
    my_model.train()

    if USE_AMP and device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    print("Start Training...")
    total_updates = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    progress_bar = tqdm(range(total_updates))
    update_step = 0

    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast_ctx:
                outputs = my_model(**batch)
                loss = outputs["loss"] / GRADIENT_ACCUMULATION

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION)
                progress_bar.update(1)
                update_step += 1

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(my_model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
