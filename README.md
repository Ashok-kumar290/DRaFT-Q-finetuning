# DRaFT-Q: Efficient Fine-Tuning with Rank-Adaptive Quantization

This repository provides DRaFT-Q adapter weights and training code for fine-tuning large language models on a text classification task.

## 🧠 What is DRaFT-Q?

**DRaFT-Q (Dynamic Rank and Fine-Tuned Quantization)** is an efficient adaptation method that:
- Dynamically adjusts adapter rank across layers
- Optimizes for memory efficiency
- Preserves model accuracy using 4-bit quantization

It is particularly suited for low-resource hardware like Colab GPUs.

## 🏋️‍♂️ Models Trained with DRaFT-Q

| Model         | Params | Dataset Used                  | Purpose              |
|---------------|--------|-------------------------------|----------------------|
| Falcon-RW-1B  | 1.3B   | SQuAD v1.1                    | Q/A   |
| LLaMA-2-7B    | 7B     | NewsSummaryMore               | Text classification  |

Both models were trained using the DRaFT-Q method with LoRA adapters.

## 📂 Contents

- `falcon-rw-1b/`: Adapter files (`adapter_model.bin`, `adapter_config.json`)
- `llama-2-7b/`: Adapter files
- `Benchmarks.ipynb`: End-to-end training & evaluation in Google Colab

## 📜 License

This project is licensed under the MIT License.
