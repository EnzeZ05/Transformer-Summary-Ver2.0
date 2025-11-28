# Handwritten GPT-2 Large Summarizer

A project where I “dig GPT-2 Large’s brain (its weights) out” of the official Hugging Face model and plug it into my own handwritten Transformer (`ModernDecoder`), then:

- load `gpt2-large` weights from Hugging Face  
- train the transplanted brain on an English summarization dataset  
- generate summaries in three decoding modes: `safe`, `balanced`, and `creative`


---

## Files

- `transformer.py` – handwritten GPT-2 decoder (multi-head attention + GELU MLP)
- `train.py` – load `gpt2-large`, copy weights into `ModernDecoder`, fine-tune on `english_train.jsonl`
- `test.py` – load `my-gpt2large-handwritten` and run summarization
- `requirements.txt` – dependencies

---

## Install

```bash
git clone https://github.com/EnzeZ05/Transformer-Summary-Ver2.0.git
cd Transformer-Summary-Ver2.0
pip install -r requirements.txt
```

---

## Example

### Input:

Hitogashima is a fictional island featured in the anime Summer Time Rendering.
Shinpei Ajiro returns to the island for the funeral of his childhood friend Ushio Kofune.
However, he discovers that Ushio's death might not have been an accident.
Legends of "Shadows" mimicking humans circulate on the island.
Shinpei encounters a Shadow that looks exactly like Ushio, and he gets shot in the head.
Instead of dying, he travels back in time to the day he arrived on the island.

### ouput:

SAFE SUMMARY:
A boy named Shinpei Ajiro arrives on a remote Japanese island to find out why his childhood friend died.

BALANCED SUMMARY:
A boy named Shinpei Ajiro has returned from the future to find out why his best friend died.

CREATIVE SUMMARY:
The characters from the Japanese horror film Summer Time Rendering travel through time at the behest of a mysterious entity called a Shadows.

---

## Liscence
MIT
