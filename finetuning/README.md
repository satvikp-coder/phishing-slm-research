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

---

## Arm A Empirical Results (Completed)

* **Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (Peak VRAM: 7.05 GB)
* **Training Time:** 16,603.5 sec (4.61 hours, 3 full epochs)
* **Evaluation Split:** `phishing_val.csv` ($n = 2,954$ emails)

| Metric | Score | Performance Context |
|---|:---:|---|
| **Accuracy** | **99.15%** | 2,929 / 2,954 correct predictions |
| **Precision** | **98.80%** | Only 15 false alarms out of 1,706 legitimate emails |
| **Recall** | **99.20%** | Caught 1,238 out of 1,248 phishing attacks |
| **F1-Score** | **0.9900** | Balanced, enterprise-grade classification |
| **AUROC** | **0.9916** | High discrimination threshold robustness |
| **False Positive Rate (FPR)** | **0.88%** | 60% lower than classical baselines (2.17%) |
| **False Negative Rate (FNR)** | **0.80%** | Threat miss rate < 1% |
| **Parse Failure Rate** | **0.00%** | 100% compliant outputs (0 failures) |

### Validation Confusion Matrix:
```
                      Predicted Legitimate (0)    Predicted Phishing (1)
Actual Legitimate (0)          1,691 (TN)                   15 (FP)
Actual Phishing (1)               10 (FN)                1,238 (TP)
```

* Artifacts: [`eval/results/preds_qwen2_5_1_5b_naive_val.csv`](../eval/results/preds_qwen2_5_1_5b_naive_val.csv) & [`finetuning/checkpoints/Qwen2.5-1.5B-Instruct-naive/`](checkpoints/Qwen2.5-1.5B-Instruct-naive/)

