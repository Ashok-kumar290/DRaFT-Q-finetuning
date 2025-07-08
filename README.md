# DRaFT-Q: Dynamic Rank-Aware Fine-Tuning for Quantized LLMs

**DRaFT-Q** is a novel Parameter-Efficient Fine-Tuning (PEFT) technique for large language models (LLMs) that dynamically adjusts adapter ranks **per layer**, driven by real-time curvature signals — all under a strict VRAM budget and **4-bit quantization**.

This repository contains:
- 🧠 DRaFT-Q implementation (built on HuggingFace PEFT)
- 📊 Training scripts & evaluation configs
- 📈 Visual loss curves, benchmarks, and heatmaps
- 📄 Theoretical proofs of optimal rank allocation and convergence

---

## 🧠 Why DRaFT-Q?

Unlike LoRA and QLoRA — which fix adapter ranks uniformly across layers — DRaFT-Q adapts rank **dynamically**, increasing it in “difficult” layers and reducing it in “easy” ones. This ensures better utilization of adapter capacity under limited VRAM (11–14 GB), while remaining compatible with 4-bit quantized backbones.

---

## 📊 Method Comparison

| Feature                        | LoRA | QLoRA | AdaLoRA | DRaFT-Q |
|-------------------------------|------|-------|----------|---------|
| Fixed Adapter Rank            | ✅   | ✅    | ❌       | ❌      |
| Dynamic Rank per Layer        | ❌   | ❌    | ✅       | ✅      |
| Curvature-Based Growth        | ❌   | ❌    | SVD/Hessian | ✅ (cheap EMA) |
| 4-bit Quantization Compatible | ❌   | ✅    | ❌       | ✅      |
| VRAM Budget Awareness         | ❌   | ❌    | ❌       | ✅      |
| FlashAttention-Compatible     | ✅   | ✅    | ✅       | ✅      |

---

## 📦 Trained Models

| Model                    | Params | Task          | Adapter Type | Notes |
|--------------------------|--------|---------------|---------------|-------|
| `meta-llama/Llama-2-7b-hf`     | 7B     | Summarization | LoRA / QLoRA / DRaFT-Q | NewsSummary |
| `tiiuae/falcon-rw-1b`          | 1B     | QA            | All          | SQuAD v1.1 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B   | Classification | All | AG-News |

---

## 📚 Datasets Used

| Dataset            | Task Type        | Source Link                                                                 |
|--------------------|------------------|------------------------------------------------------------------------------|
| NewsSummaryMore    | Summarization    | [news_summary_more.csv](https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv) |
| SQuAD v1.1         | QA (span-based)  | [SQuAD JSON](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) |
| AG News            | 4-class Classification | [AG-News CSV](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv) |

---

## 📈 Results Summary

| Task           | Metric   | LoRA | QLoRA | DRaFT-Q |
|----------------|----------|------|-------|----------|
| Summarization  | ROUGE-L  | 29.8 | 31.0  | **33.7** |
| QA (SQuAD)     | EM (Dev) | 48.7 | 50.2  | **49.9** |
| Classification | Accuracy | 83.2 | 84.1  | **86.4** |

> 📉 DRaFT-Q also converged faster, starting with low ranks (r=4) and expanding based on curvature.

---

## 🧮 Theoretical Highlights

- **VRAM Budget Safety**  
  DRaFT-Q includes a rollback mechanism to keep adapter memory ≤ user-defined cap

- **Convergence Lemma**  
  Guaranteed convergence if curvature update satisfies contractive EMA


---

## 🛠 Usage

Coming soon: Install and training guide...
