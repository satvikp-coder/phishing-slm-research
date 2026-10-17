# Fine-Tuning Directory (`finetuning/`)

This directory contains the core experimental training pipelines for fine-tuning Small Language Models (SLMs) using Parameter-Efficient Fine-Tuning (PEFT / QLoRA).

---

## Directory Organization

```
finetuning/
├── Phishing_SLM_QLoRA_Starter.ipynb # End-to-end QLoRA training and inference notebook (Colab T4 compatible)
├── checkpoints/                    # Saved LoRA adapter weights (.gitkeep; checkpoints stored on Drive)
└── README.md                       # This guide
```

---

## The Core Research Experiment (Weeks 5 & 6)

The project investigates two distinct fine-tuning arms under strictly controlled, identical conditions:

| Parameter | Arm A: Naive LoRA (Week 5) | Arm B: CoT-Distilled LoRA (Week 6) |
|---|---|---|
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Quantization** | 4-bit NormalFloat (`bitsandbytes`, NF4) | 4-bit NormalFloat (`bitsandbytes`, NF4) |
| **LoRA Rank ($r$)** | $r = 16$ | $r = 16$ |
| **LoRA Alpha ($\alpha$)** | $\alpha = 32$ | $\alpha = 32$ |
| **Target Modules** | All linear attention projections | All linear attention projections |
| **Learning Rate** | $2 \times 10^{-4}$ with cosine decay | $2 \times 10^{-4}$ with cosine decay |
| **Global Seed** | `42` | `42` |
| **Target Output Format** | `Phishing` or `Legitimate` (direct label) | `[Reasoning chain] -> Label` |

---

## Fair Comparison Guarantee

To maintain strict scientific validity:
* Both arms use the exact same 20,681 training emails (`phishing_train.csv`).
* Loss is computed **strictly on completion tokens** (`DataCollatorForCompletionOnlyLM`), not the input prompt.
* Input emails are pre-truncated to 800 tokens to prevent out-of-memory errors on free Colab T4 GPUs.
* Model checkpoints are saved directly to Google Drive so progress is never lost during disconnects.
