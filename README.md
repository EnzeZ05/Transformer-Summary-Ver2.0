# Handwritten GPT-2 Large Summarizer

A from-scratch PyTorch implementation of a GPT-2-style decoder (`ModernDecoder`) plus scripts to:

- load **gpt2-large** weights from Hugging Face  
- fine-tune on an English summarization dataset  
- generate summaries in different decoding modes (`safe / balanced / creative`)

---

## Files

- `transformer.py` – handwritten GPT-2 decoder (multi-head attention + GELU MLP)
- `train.py` – load `gpt2-large`, copy weights into `ModernDecoder`, fine-tune on `english_train.jsonl`
- `test.py` – load `my-gpt2large-handwritten` and run summarization
- `my-gpt2large-handwritten/` – fine-tuned weights + tokenizer
- `english_train.jsonl`, `english_val.jsonl`, `english_test.jsonl` – JSONL summarization data
- `requirements.txt` – dependencies

---

## Install

```bash
git clone https://github.com/EnzeZ05/autod1l-tmp1.git
cd autod1l-tmp1
pip install -r requirements.txt

## Example

Input:
Hitogashima is a fictional island featured in the anime Summer Time Rendering.
Shinpei Ajiro returns to the island for the funeral of his childhood friend Ushio Kofune.
However, he discovers that Ushio's death might not have been an accident.
Legends of "Shadows" mimicking humans circulate on the island.
Shinpei encounters a Shadow that looks exactly like Ushio, and he gets shot in the head.
Instead of dying, he travels back in time to the day he arrived on the island.

ouput:
==============================
SAFE SUMMARY:
A boy named Shinpei Ajiro arrives on a remote Japanese island to find out why his childhood friend died.
==============================

==============================
BALANCED SUMMARY:
The protagonist from the Japanese anime series "Summer Time Rendering", who arrives at the island to attend a funeral, finds himself transported back in time to the day he was born.
==============================

==============================
CREATIVE SUMMARY:
After being shot in the head by a mysterious person, a high school student named Shinpei Ajiro returns home to find his mother dead. He's forced to return to the island where he was born before he can save her life, and must fight off waves of shadow creatures while trying to keep up with his new friends and classmates.
==============================

## Liscence
MIT
