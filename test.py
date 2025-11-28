import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HOME", "/root/autodl-fs/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-fs/hf_cache")

import contextlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import transformer

CKPT_DIR = "summarizer"
BASE_MODEL_NAME = "gpt2-large"

D_MODEL = 1280
NUM_LAYERS = 36
NUM_HEADS = 20
D_FF = 5120


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_path = os.path.join(CKPT_DIR, "pytorch_model.bin")
    print(f"Loading weights from {model_path}...")

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    if "pos_emb.weight" in state_dict:
        max_len = state_dict["pos_emb.weight"].shape[0]
        print(f"Detected max_len from checkpoint: {max_len}")
    else:
        max_len = 1024
        print(f"pos_emb not found in state_dict, fallback max_len={max_len}")

    print("Initializing Custom ModernDecoder (GPT-2 Large Specs)...")
    model = transformer.ModernDecoder(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        layers=NUM_LAYERS,
        heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=max_len,
        dropout=0.0,
        token_id=tokenizer.pad_token_id,
    ).to(device)

    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Load status: {msg}")

    model.eval()
    if device.type == "cuda":
        model.to(dtype=torch.bfloat16)

    return model, tokenizer, device


@torch.inference_mode()
def generate(
        model,
        tokenizer,
        device,
        text: str,
        mode: str = "balanced",
        max_new_tokens: int = 512,
):
    if mode == "safe":
        temperature = 0.10
        top_p = 0.5
        repetition_penalty = 1.15
        prompt = (
            "You are a careful assistant that writes accurate, faithful summaries.\n"
            "Write a concise English summary of the passage below.\n"
            "Do not invent any events that are not explicitly mentioned.\n\n"
            f"Passage:\n{text}\n\n"
            "Summary:"
        )
    elif mode == "creative":
        temperature = 0.7
        top_p = 0.95
        repetition_penalty = 1.2
        prompt = (
            "You are a storyteller assistant. Read the passage and write a short, vivid English summary.\n"
            "You may slightly rephrase or combine details, but the plot should stay consistent with the passage.\n\n"
            f"Passage:\n{text}\n\n"
            "Summary:"
        )
    else:
        temperature = 0.5
        top_p = 0.9
        repetition_penalty = 1.2
        prompt = (
            "You are a helpful assistant that writes concise, clear summaries.\n"
            "Write a short English summary of the following passage.\n\n"
            f"Passage:\n{text}\n\n"
            "Summary:"
        )

    if hasattr(model, "pos_emb"):
        max_seq_len = model.pos_emb.weight.size(0)
    else:
        max_seq_len = 1024

    prompt_max_tokens = max_seq_len - max_new_tokens - 8
    if prompt_max_tokens <= 0:
        raise ValueError(f"max_new_tokens={max_new_tokens} is too large for max_seq_len={max_seq_len}")

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attn_mask = (input_ids != pad_id).long().to(device)

    if device.type == "cuda":
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = contextlib.nullcontext()

    generated = input_ids
    window_size = max_seq_len

    for _ in range(max_new_tokens):
        seq_len = generated.size(1)
        if seq_len >= max_seq_len:
            break

        if seq_len > window_size:
            context = generated[:, -window_size:]
            context_mask = attn_mask[:, -window_size:]
        else:
            context = generated
            context_mask = attn_mask

        with ctx:
            outputs = model(
                input_ids=context,
                attention_mask=context_mask,
            )
            logits = outputs["logits"][:, -1, :].float()

        if repetition_penalty > 1.0:
            recent_tokens = generated[0, max(0, seq_len - 256):].tolist()
            for tid in set(recent_tokens):
                logits[0, tid] /= repetition_penalty

        logits = logits / max(temperature, 1e-5)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), device=device, dtype=attn_mask.dtype)],
            dim=1,
        )

    output_tokens = generated[0][prompt_len:]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return output_text.strip()


if __name__ == "__main__":
    text_path = "summer_time.txt"

    if not os.path.exists(text_path):
        article = """
        Hitogashima is a fictional island featured in the anime Summer Time Rendering. 
        Shinpei Ajiro returns to the island for the funeral of his childhood friend Ushio Kofune. 
        However, he discovers that Ushio's death might not have been an accident. 
        Legends of "Shadows" mimicking humans circulate on the island.
        Shinpei encounters a Shadow that looks exactly like Ushio, and he gets shot in the head.
        Instead of dying, he travels back in time to the day he arrived on the island.
        """
    else:
        with open(text_path, "r", encoding="utf-8") as f:
            article = f.read()

    print(f"Original Text Length: {len(article)} chars")

    model, tok, dev = load_model()

    for mode in ["safe", "balanced", "creative"]:
        print("\n" + "=" * 30)
        print(f"{mode.upper()} SUMMARY:")
        summary = generate(model, tok, dev, article, mode=mode, max_new_tokens=256)
        print(summary)
        print("=" * 30)
