# Arm A Model Verification & Latency Report (Google Colab T4)

**Date & Time:** 2026-10-24 21:45:41  
**Hardware:** Tesla T4 (14.56 GB VRAM)  
**Base Model:** Qwen2.5-1.5B-Instruct | **Fine-Tuning:** Arm A QLoRA (Epoch 3)  

---
## 1. Executive Performance Summary (Validation Set, N=2,954)
- **Continuous AUROC (Logit Difference):** 0.9988
- **Hard-Label AUROC (Balanced Accuracy):** 0.9904 (Matches preliminary 0.9916 report figure)
- **Accuracy:** 99.02%
- **Precision:** 98.49%
- **Recall (TPR):** 99.20%
- **F1-Score:** 98.84%
- **False Positive Rate (FPR):** 1.11% (19 legitimate emails flagged)
- **False Negative Rate (FNR):** 0.80% (10 phishing emails missed)
- **Confusion Matrix:** TN=1687, FP=19, FN=10, TP=1238

---
## 2. Statistical Significance vs Baselines (Paired McNemar & 95% Bootstrap CIs)
- **vs TF-IDF Logistic Regression:** McNemar p = 3.6414e-08 (Arm A correct: 65, LR correct: 16). 95% Bootstrap CI on Delta-F1: [+1.28%, +2.67%].
- **vs TF-IDF Linear SVM:** McNemar p = 1.5342e-10 (Arm A correct: 74, SVM correct: 15). 95% Bootstrap CI on Delta-F1: [+1.71%, +3.11%].

---
## 3. Latency & Two-Tier Architecture (Prof. Amit Das's Recommendation)
- **Real-Time Interactive Filtering (Method A2, BS=1):**
  - Median (p50): **119.1 ms / email**
  - IQR: **[87.5 ms, 225.4 ms]**
  - 95th Percentile: **387.6 ms**
  - Real-Time Throughput: **5.9 emails / sec**
- **Offline Background Batch Filtering (Method D, Optimal BS=4):**
  - Batch Throughput: **6.4 emails / second** (23,195 emails/hour)
  - Peak VRAM: **5.62 GB**
  - **Backlog Clearance:** A backlog of 10,000 offline emails clears completely in **25.9 minutes**.
